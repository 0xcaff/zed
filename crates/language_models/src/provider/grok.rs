use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use credentials_provider::CredentialsProvider;
use editor::{Editor, EditorElement, EditorStyle};
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use gpui::{
    AnyView, App, AsyncApp, Context, Entity, FontStyle, Task, TextStyle, WhiteSpace, Window,
    prelude::*, relative, rems,
};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, RateLimiter,
};
use menu::Confirm;
use open_ai::{ResponseStreamEvent, stream_completion};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use theme::ThemeSettings;
use ui::{Button, Color, IconName, Label, LabelCommon, LabelSize, List, prelude::*};
use util::ResultExt;

use crate::AllLanguageModelSettings;
use crate::provider::open_ai::{OpenAiEventMapper, count_open_ai_tokens, into_open_ai};
use crate::ui::InstructionListItem;

const PROVIDER_ID: &str = "grok";
const PROVIDER_NAME: &str = "Grok";
const XAI_API_KEY_VAR: &str = "XAI_API_KEY";

#[derive(Clone, Default)]
pub struct GrokSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub display_name: String,
    pub max_tokens: usize,
    pub max_output_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
}

pub struct GrokLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Entity<State>,
}

pub struct State {
    api_key: Option<String>,
    api_key_from_env: bool,
    _subscription: gpui::Subscription,
}

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    fn reset_api_key(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        cx.spawn(async move |this, cx| {
            this.update(cx, |this, cx| {
                this.api_key = None;
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn set_api_key(&mut self, api_key: String, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = AllLanguageModelSettings::get_global(cx)
            .grok
            .api_url
            .clone();
        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(&api_url, "Bearer", api_key.as_bytes(), &cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn authenticate(&self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = AllLanguageModelSettings::get_global(cx)
            .grok
            .api_url
            .clone();
        cx.spawn(async move |this, cx| {
            let (api_key, from_env) = if let Ok(api_key) = std::env::var(XAI_API_KEY_VAR) {
                (api_key, true)
            } else {
                let (_, api_key) = credentials_provider
                    .read_credentials(&api_url, &cx)
                    .await?
                    .ok_or(AuthenticateError::CredentialsNotFound)?;
                (
                    String::from_utf8(api_key).context("invalid Grok API key")?,
                    false,
                )
            };

            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = from_env;
                cx.notify();
            })?;

            Ok(())
        })
    }
}

impl GrokLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| State {
            api_key: None,
            api_key_from_env: false,
            _subscription: cx.observe_global::<SettingsStore>(|_this: &mut State, cx| {
                cx.notify();
            }),
        });

        Self { http_client, state }
    }

    fn create_language_model(&self, model: open_ai::Model) -> Arc<dyn LanguageModel> {
        Arc::new(GrokLanguageModel {
            id: LanguageModelId::from(model.id().to_string()),
            model,
            state: self.state.clone(),
            http_client: self.http_client.clone(),
            request_limiter: RateLimiter::new(4),
        })
    }
}

impl LanguageModelProviderState for GrokLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

impl LanguageModelProvider for GrokLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn icon(&self) -> IconName {
        IconName::ZedAssistant
    }

    fn default_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(open_ai::Model::Custom {
            name: "grok-3-latest".to_string(),
            display_name: Some("Grok 3".to_string()),
            max_tokens: 131072,
            max_output_tokens: Some(4096),
            max_completion_tokens: Some(4096),
        }))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(open_ai::Model::Custom {
            name: "grok-3-fast-latest".to_string(),
            display_name: Some("Grok 3 Fast".to_string()),
            max_tokens: 131072,
            max_output_tokens: Some(4096),
            max_completion_tokens: Some(4096),
        }))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add Grok 3 models (latest only)
        models.insert(
            "grok-3-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-latest".to_string(),
                display_name: Some("Grok 3".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );
        
        models.insert(
            "grok-3-fast-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-fast-latest".to_string(),
                display_name: Some("Grok 3 Fast".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        // Add Grok 3 Mini models (non-thinking variants)
        models.insert(
            "grok-3-mini-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-mini-latest".to_string(),
                display_name: Some("Grok 3 Mini".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );
        
        models.insert(
            "grok-3-mini-fast-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-mini-fast-latest".to_string(),
                display_name: Some("Grok 3 Mini Fast".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        // Add Grok 3 Mini models (thinking variants)
        models.insert(
            "grok-3-mini-thinking-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-mini-latest".to_string(),
                display_name: Some("Grok 3 Mini (Thinking)".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );
        
        models.insert(
            "grok-3-mini-thinking-fast-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-3-mini-fast-latest".to_string(),
                display_name: Some("Grok 3 Mini Fast (Thinking)".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        // Add Grok 2 models (latest only)
        models.insert(
            "grok-2-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-2-latest".to_string(),
                display_name: Some("Grok 2".to_string()),
                max_tokens: 131072,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        // Add Grok 2 Vision models (latest only)
        models.insert(
            "grok-2-vision-latest".to_string(),
            open_ai::Model::Custom {
                name: "grok-2-vision-latest".to_string(),
                display_name: Some("Grok 2 Vision".to_string()),
                max_tokens: 32768,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        // Override with available models from settings if any
        for model in &AllLanguageModelSettings::get_global(cx)
            .grok
            .available_models
        {
            models.insert(
                model.name.clone(),
                open_ai::Model::Custom {
                    name: model.name.clone(),
                    display_name: Some(model.display_name.clone()),
                    max_tokens: model.max_tokens,
                    max_output_tokens: model.max_output_tokens,
                    max_completion_tokens: model.max_completion_tokens,
                },
            );
        }

        models
            .into_values()
            .map(|model| self.create_language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(&self, window: &mut Window, cx: &mut App) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<Result<()>> {
        self.state.update(cx, |state, cx| state.reset_api_key(cx))
    }
}

pub struct GrokLanguageModel {
    id: LanguageModelId,
    model: open_ai::Model,
    state: gpui::Entity<State>,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
}

impl GrokLanguageModel {
    fn stream_completion(
        &self,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>>>
    {
        let http_client = self.http_client.clone();
        let Ok((api_key, api_url)) = cx.read_entity(&self.state, |state, cx| {
            let settings = &AllLanguageModelSettings::get_global(cx).grok;
            (state.api_key.clone(), settings.api_url.clone())
        }) else {
            return futures::future::ready(Err(anyhow!("App state dropped"))).boxed();
        };

        let future = self.request_limiter.stream(async move {
            let api_key = api_key.context("Missing Grok API Key")?;
            let api_url = if api_url.is_empty() {
                "https://api.x.ai/v1".to_string()
            } else {
                api_url
            };
            let request = stream_completion(http_client.as_ref(), &api_url, &api_key, request);
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

impl LanguageModel for GrokLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        LanguageModelProviderId(PROVIDER_ID.into())
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        LanguageModelProviderName(PROVIDER_NAME.into())
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        // Only Grok 2 Vision models support images
        self.model.id().contains("vision")
    }

    fn supports_tool_choice(&self, choice: LanguageModelToolChoice) -> bool {
        match choice {
            LanguageModelToolChoice::Auto => true,
            LanguageModelToolChoice::Any => true,
            LanguageModelToolChoice::None => true,
        }
    }

    fn telemetry_id(&self) -> String {
        format!("grok/{}", self.model.id())
    }

    fn max_token_count(&self) -> usize {
        self.model.max_token_count()
    }

    fn max_output_tokens(&self) -> Option<u32> {
        self.model.max_output_tokens()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        cx: &App,
    ) -> BoxFuture<'static, Result<usize>> {
        count_open_ai_tokens(request, self.model.clone(), cx)
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<
                'static,
                Result<LanguageModelCompletionEvent, LanguageModelCompletionError>,
            >,
        >,
    > {
        let request = into_open_ai(request, &self.model, self.max_output_tokens());
        let completions = self.stream_completion(request, cx);
        async move {
            let mapper = OpenAiEventMapper::new();
            let stream = mapper.map_stream(completions.await?);
            Ok(stream.boxed())
        }
        .boxed()
    }
}

struct ConfigurationView {
    api_key_editor: Entity<Editor>,
    state: gpui::Entity<State>,
    load_credentials_task: Option<Task<()>>,
}

impl ConfigurationView {
    fn new(state: gpui::Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| {
            let mut editor = Editor::single_line(window, cx);
            editor.set_placeholder_text("xai-000000000000000000000000000000000000000000000000", cx);
            editor
        });

        cx.observe(&state, |_, _, cx| {
            cx.notify();
        })
        .detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                if let Some(task) = state
                    .update(cx, |state, cx| state.authenticate(cx))
                    .log_err()
                {
                    // We don't log an error, because "not signed in" is also an error.
                    let _ = task.await;
                }

                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .log_err();
            }
        }));

        Self {
            api_key_editor,
            state,
            load_credentials_task,
        }
    }

    fn save_api_key(&mut self, _: &Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if api_key.is_empty() {
            return;
        }

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(api_key, cx))?
                .await
        })
        .detach_and_log_err(cx);

        cx.notify();
    }

    fn render_api_key_editor(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let settings = ThemeSettings::get_global(cx);
        let text_style = TextStyle {
            color: cx.theme().colors().text,
            font_family: settings.ui_font.family.clone(),
            font_features: settings.ui_font.features.clone(),
            font_fallbacks: settings.ui_font.fallbacks.clone(),
            font_size: rems(0.875).into(),
            font_weight: settings.ui_font.weight,
            font_style: FontStyle::Normal,
            line_height: relative(1.3),
            white_space: WhiteSpace::Normal,
            ..Default::default()
        };
        EditorElement::new(
            &self.api_key_editor,
            EditorStyle {
                background: cx.theme().colors().editor_background,
                local_player: cx.theme().players().local(),
                text: text_style,
                ..Default::default()
            },
        )
    }

    fn should_render_editor(&self, cx: &Context<Self>) -> bool {
        !self.state.read(cx).api_key_from_env
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let env_var_set = self.state.read(cx).api_key_from_env;

        if self.load_credentials_task.is_some() {
            div().child(Label::new("Loading credentials...")).into_any()
        } else if self.should_render_editor(cx) {
            v_flex()
                .size_full()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new("To use Zed's assistant with Grok, you need to add an API key. Follow these steps:"))
                .child(
                    List::new()
                        .child(InstructionListItem::new(
                            "Create one by visiting",
                            Some("X.AI's console"),
                            Some("https://x.ai/"),
                        ))
                        .child(InstructionListItem::text_only(
                            "Ensure your X.AI account has credits",
                        ))
                        .child(InstructionListItem::text_only(
                            "Paste your API key below and hit enter to start using the assistant",
                        )),
                )
                .child(
                    h_flex()
                        .w_full()
                        .my_2()
                        .px_2()
                        .py_1()
                        .bg(cx.theme().colors().editor_background)
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .rounded_sm()
                        .child(self.render_api_key_editor(cx)),
                )
                .child(
                    Label::new(
                        format!("You can also assign the {XAI_API_KEY_VAR} environment variable and restart Zed."),
                    )
                    .size(LabelSize::Small).color(Color::Muted),
                )
                .into_any()
        } else {
            v_flex()
                .child(
                    Label::new(if env_var_set {
                        format!("You're using the Grok API key from the {XAI_API_KEY_VAR} environment variable.")
                    } else {
                        "You're authenticated with the Grok API.".to_string()
                    })
                )
                .into_any()
        }
    }
}
