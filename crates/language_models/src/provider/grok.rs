use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context as _, Result, anyhow};
use futures::future::BoxFuture;
use futures::{FutureExt, StreamExt};
use gpui::{AnyView, App, AsyncApp, Context, Task, Window, prelude::*};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionError, LanguageModelCompletionEvent,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    LanguageModelToolChoice, RateLimiter,
};
use open_ai::{ResponseStreamEvent, stream_completion};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use ui::{Color, IconName, Label, LabelCommon, LabelSize, prelude::*};

use crate::AllLanguageModelSettings;
use crate::provider::open_ai::{OpenAiEventMapper, count_open_ai_tokens, into_open_ai};

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
        self.api_key = Some(api_key);
        self.api_key_from_env = false;
        cx.notify();
        Task::ready(Ok(()))
    }

    fn authenticate(&self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        cx.spawn(async move |this, cx| {
            let (api_key, from_env) = if let Ok(api_key) = std::env::var(XAI_API_KEY_VAR) {
                (api_key, true)
            } else {
                return Err(AuthenticateError::CredentialsNotFound);
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
            name: "grok-3".to_string(),
            display_name: Some("Grok-3".to_string()),
            max_tokens: 128000,
            max_output_tokens: Some(4096),
            max_completion_tokens: Some(4096),
        }))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<Arc<dyn LanguageModel>> {
        Some(self.create_language_model(open_ai::Model::Custom {
            name: "grok-2".to_string(),
            display_name: Some("Grok-2".to_string()),
            max_tokens: 128000,
            max_output_tokens: Some(4096),
            max_completion_tokens: Some(4096),
        }))
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = BTreeMap::default();

        // Add default Grok models
        models.insert(
            "grok-3".to_string(),
            open_ai::Model::Custom {
                name: "grok-3".to_string(),
                display_name: Some("Grok-3".to_string()),
                max_tokens: 128000,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        models.insert(
            "grok-2".to_string(),
            open_ai::Model::Custom {
                name: "grok-2".to_string(),
                display_name: Some("Grok-2".to_string()),
                max_tokens: 128000,
                max_output_tokens: Some(4096),
                max_completion_tokens: Some(4096),
            },
        );

        models.insert(
            "grok-1".to_string(),
            open_ai::Model::Custom {
                name: "grok-1".to_string(),
                display_name: Some("Grok-1".to_string()),
                max_tokens: 128000,
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

    fn configuration_view(&self, _window: &mut Window, cx: &mut App) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), cx))
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
        false
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
    state: gpui::Entity<State>,
    load_credentials_task: Option<Task<Result<(), AuthenticateError>>>,
}

impl ConfigurationView {
    fn new(state: gpui::Entity<State>, cx: &mut Context<Self>) -> Self {
        let load_credentials_task = state.update(cx, |state, cx| state.authenticate(cx));

        Self {
            state,
            load_credentials_task: Some(load_credentials_task),
        }
    }
}

impl Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let is_authenticated = self.state.read(cx).is_authenticated();
        let has_env_key = self.state.read(cx).api_key_from_env;

        v_flex()
            .p_4()
            .w_full()
            .child(
                h_flex().child(
                    Label::new("Grok")
                        .size(LabelSize::Large),
                ),
            )
            .child(
                h_flex()
                    .my_2()
                    .child(
                        Label::new(if is_authenticated {
                            "✓ Authenticated"
                        } else {
                            "✗ Not authenticated"
                        })
                        .color(if is_authenticated {
                            Color::Success
                        } else {
                            Color::Error
                        }),
                    ),
            )
            .child(if has_env_key {
                v_flex()
                    .child(
                        Label::new(format!(
                            "Using API key from {} environment variable",
                            XAI_API_KEY_VAR
                        ))
                        .color(Color::Muted),
                    )
                    .into_any_element()
            } else {
                v_flex()
                    .child(
                        Label::new("Please set your XAI_API_KEY environment variable to authenticate.")
                            .color(Color::Muted)
                    )
                    .into_any_element()
            })
            .child(
                v_flex()
                    .mt_4()
                    .child(
                        Label::new("Grok is compatible with the OpenAI API. You can get your API key from https://x.ai/.")
                            .color(Color::Muted)
                            .size(LabelSize::Small)
                    )
            )
    }
}
