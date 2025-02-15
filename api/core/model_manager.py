# model_manager.py
import logging
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import IO, Any, Optional, Union, cast

from configs import dify_config
from core.entities.embedding_type import EmbeddingInputType
from core.entities.provider_configuration import ProviderConfiguration, ProviderModelBundle
from core.entities.provider_entities import ModelLoadBalancingConfiguration
from core.errors.error import ProviderTokenNotInitError
from core.model_runtime.callbacks.base_callback import Callback
from core.model_runtime.entities.llm_entities import LLMResult
from core.model_runtime.entities.message_entities import PromptMessage, PromptMessageTool
from core.model_runtime.entities.model_entities import ModelType
from core.model_runtime.entities.rerank_entities import RerankResult
from core.model_runtime.entities.text_embedding_entities import TextEmbeddingResult
from core.model_runtime.errors.invoke import InvokeAuthorizationError, InvokeConnectionError, InvokeRateLimitError
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.model_runtime.model_providers.__base.moderation_model import ModerationModel
from core.model_runtime.model_providers.__base.rerank_model import RerankModel
from core.model_runtime.model_providers.__base.speech2text_model import Speech2TextModel
from core.model_runtime.model_providers.__base.text_embedding_model import TextEmbeddingModel
from core.model_runtime.model_providers.__base.tts_model import TTSModel
from core.provider_manager import ProviderManager
from extensions.ext_redis import redis_client
from models.provider import ProviderType
# from utils.encrypter import encrypter  <-- Remove this
from core.helper.encrypter import get_decrypt_decoding, decrypt_token_with_decoding  # Correct import

logger = logging.getLogger(__name__)


class ModelInstance:
    """
    Model instance class
    """

    def __init__(self, provider_model_bundle: ProviderModelBundle, model: str) -> None:
        self.provider_model_bundle = provider_model_bundle
        self.model = model
        self.provider = provider_model_bundle.configuration.provider.provider
        self.credentials = self._fetch_credentials_from_bundle(provider_model_bundle, model)
        self.model_type_instance = self.provider_model_bundle.model_type_instance
        self.load_balancing_manager = self._get_load_balancing_manager(
            configuration=provider_model_bundle.configuration,
            model_type=provider_model_bundle.model_type_instance.model_type,
            model=model,
            credentials=self.credentials,
        )

    @staticmethod
    def _fetch_credentials_from_bundle(provider_model_bundle: ProviderModelBundle, model: str) -> dict:
        """
        Fetch credentials from provider model bundle
        """
        configuration = provider_model_bundle.configuration
        model_type = provider_model_bundle.model_type_instance.model_type
        credentials = configuration.get_current_credentials(model_type=model_type, model=model)

        if credentials is None:
            raise ProviderTokenNotInitError(f"Model {model} credentials is not initialized.")

        return credentials

    @staticmethod
    def _get_load_balancing_manager(
        configuration: ProviderConfiguration, model_type: ModelType, model: str, credentials: dict
    ) -> Optional["LBModelManager"]:
        """
        Get load balancing model credentials
        """
        if configuration.model_settings and configuration.using_provider_type == ProviderType.CUSTOM:
            current_model_setting = None
            # check if model is disabled by admin
            for model_setting in configuration.model_settings:
                if model_setting.model_type == model_type and model_setting.model == model:
                    current_model_setting = model_setting
                    break

            # check if load balancing is enabled
            if current_model_setting and current_model_setting.load_balancing_configs:
                # use load balancing proxy to choose credentials
                lb_model_manager = LBModelManager(
                    tenant_id=configuration.tenant_id,
                    provider=configuration.provider.provider,
                    model_type=model_type,
                    model=model,
                    load_balancing_configs=current_model_setting.load_balancing_configs,
                    managed_credentials=credentials if configuration.custom_configuration.provider else None,
                )

                return lb_model_manager

        return None

    def invoke_llm(
        self,
        prompt_messages: Sequence[PromptMessage],
        model_parameters: Optional[dict] = None,
        tools: Sequence[PromptMessageTool] | None = None,
        stop: Optional[Sequence[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model
        """
        if not isinstance(self.model_type_instance, LargeLanguageModel):
            raise Exception("Model type instance is not LargeLanguageModel")

        self.model_type_instance = cast(LargeLanguageModel, self.model_type_instance)
        return cast(
            Union[LLMResult, Generator],
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials,  # ADDED credentials
                prompt_messages=prompt_messages,
                model_parameters=model_parameters,
                tools=tools,
                stop=stop,
                stream=stream,
                user=user,
                callbacks=callbacks,
            ),
        )

    def get_llm_num_tokens(
        self, prompt_messages: list[PromptMessage], tools: Optional[list[PromptMessageTool]] = None
    ) -> int:
        """
        Get number of tokens for llm
        """
        if not isinstance(self.model_type_instance, LargeLanguageModel):
            raise Exception("Model type instance is not LargeLanguageModel")

        self.model_type_instance = cast(LargeLanguageModel, self.model_type_instance)
        return cast(
            int,
            self._round_robin_invoke(
                function=self.model_type_instance.get_num_tokens,
                model=self.model,
                credentials=self.credentials,  # ADDED credentials
                prompt_messages=prompt_messages,
                tools=tools,
            ),
        )

    def invoke_text_embedding(
        self, texts: list[str], user: Optional[str] = None, input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model
        """
        if not isinstance(self.model_type_instance, TextEmbeddingModel):
            raise Exception("Model type instance is not TextEmbeddingModel")

        self.model_type_instance = cast(TextEmbeddingModel, self.model_type_instance)
        return cast(
            TextEmbeddingResult,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials,  # ADDED credentials
                texts=texts,
                user=user,
                input_type=input_type,
            ),
        )

    def get_text_embedding_num_tokens(self, texts: list[str]) -> int:
        if not isinstance(self.model_type_instance, TextEmbeddingModel):
            raise Exception("Model type instance is not TextEmbeddingModel")

        self.model_type_instance = cast(TextEmbeddingModel, self.model_type_instance)
        return cast(
            int,
            self._round_robin_invoke(
                function=self.model_type_instance.get_num_tokens,
                model=self.model,
                credentials=self.credentials,  # ADDED credentials
                texts=texts,
            ),
        )
    def invoke_rerank(
            self,
            query: str,
            docs: list[str],
            score_threshold: Optional[float] = None,
            top_n: Optional[int] = None,
            user: Optional[str] = None,
    ) -> RerankResult:
        if not isinstance(self.model_type_instance, RerankModel):
            raise Exception("Model type instance is not RerankModel")

        self.model_type_instance = cast(RerankModel, self.model_type_instance)
        return cast(
            RerankResult,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials, # ADDED credentials
                query=query,
                docs=docs,
                score_threshold=score_threshold,
                top_n=top_n,
                user=user
            )
        )


    def invoke_moderation(self, text: str, user: Optional[str] = None) -> bool:
        if not isinstance(self.model_type_instance, ModerationModel):
            raise Exception("Model type instance is not ModerationModel")

        self.model_type_instance = cast(ModerationModel, self.model_type_instance)
        return cast(
            bool,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials, # ADDED credentials
                text=text,
                user=user
            )
        )
    def invoke_speech2text(self, file: IO[bytes], user: Optional[str] = None) -> str:

        if not isinstance(self.model_type_instance, Speech2TextModel):
            raise Exception("Model type instance is not Speech2TextModel")

        self.model_type_instance = cast(Speech2TextModel, self.model_type_instance)
        return cast(
            str,
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials, # ADDED credentials
                file=file,
                user=user
            )
        )

    def invoke_tts(self, content_text: str, tenant_id:str, voice: str, user: Optional[str] = None) -> Iterable[bytes]:

        if not isinstance(self.model_type_instance, TTSModel):
            raise Exception("Model type instance is not TTSModel")

        self.model_type_instance = cast(TTSModel, self.model_type_instance)
        return cast(
            Iterable[bytes],
            self._round_robin_invoke(
                function=self.model_type_instance.invoke,
                model=self.model,
                credentials=self.credentials, # ADDED credentials
                content_text=content_text,
                user=user,
                tenant_id = tenant_id,
                voice = voice
            )
        )
    def get_tts_voices(self, language: Optional[str] = None) -> list:

        if not isinstance(self.model_type_instance, TTSModel):
            raise Exception("Model type instance is not TTSModel")
        self.model_type_instance = cast(TTSModel, self.model_type_instance)
        return self.model_type_instance.get_tts_model_voices(model=self.model, credentials=self.credentials, language= language)


    def _round_robin_invoke(self, function: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Round-robin invoke with proper credential decryption.
        """
        if not self.load_balancing_manager:
            return function(*args, **kwargs)

        last_exception: Optional[
            Union[InvokeRateLimitError, InvokeAuthorizationError, InvokeConnectionError]
        ] = None

        main_credentials = self.credentials  # Use pre-fetched, decrypted credentials

        while True:
            lb_config = self.load_balancing_manager.fetch_next()
            if not lb_config:
                if last_exception:
                    raise last_exception
                else:
                    raise ProviderTokenNotInitError("No valid model credentials available.")

            try:
                if lb_config.name == "__inherit__":
                    credentials = main_credentials
                else:
                    credentials = lb_config.credentials.copy()
                    tenant_id = self.provider_model_bundle.configuration.tenant_id  # Get tenant_id
                    rsa_key, cipher_rsa = get_decrypt_decoding(tenant_id) # 直接使用导入的函数

                    for key, value in credentials.items():
                        if isinstance(value, str):
                            try:
                                if value.startswith('HYBRID:'):
                                    decrypted_value = decrypt_token_with_decoding(  # 直接使用导入的函数
                                        value, rsa_key, cipher_rsa
                                    )
                                    credentials[key] = decrypted_value
                            except Exception as e:
                                logger.warning(f"Failed to decrypt credential {key}: {e}")
                                # Keep original value on failure
                                continue
                logger.debug(f"Using LB credentials: {credentials}")
                kwargs['credentials'] = credentials
                return function(*args, **kwargs)  # Corrected call


            except InvokeRateLimitError as e:
                logger.warning(f"Rate limit error: {e}")
                self.load_balancing_manager.cooldown(lb_config, expire=60)
                last_exception = e
            except (InvokeAuthorizationError, InvokeConnectionError) as e:
                logger.warning(f"Auth/Connection error: {e}")
                self.load_balancing_manager.cooldown(lb_config, expire=10)
                last_exception = e
            except Exception as e:
                logger.exception(f"Unexpected error during invoke: {e}")  # More detailed logging
                raise


class ModelManager:
    def __init__(self) -> None:
        self._provider_manager = ProviderManager()

    def get_model_instance(self, tenant_id: str, provider: str, model_type: ModelType, model: str) -> ModelInstance:
        if not provider:
            return self.get_default_model_instance(tenant_id, model_type)

        provider_model_bundle = self._provider_manager.get_provider_model_bundle(
            tenant_id=tenant_id, provider=provider, model_type=model_type
        )
        return ModelInstance(provider_model_bundle, model)

    def get_default_provider_model_name(self, tenant_id: str, model_type: ModelType) -> tuple[str, str]:
        return self._provider_manager.get_first_provider_first_model(tenant_id, model_type)


    def get_default_model_instance(self, tenant_id: str, model_type: ModelType) -> ModelInstance:
        default_model_entity = self._provider_manager.get_default_model(tenant_id=tenant_id, model_type=model_type)
        if not default_model_entity:
            raise ProviderTokenNotInitError(f"Default model not found for {model_type}")
        return self.get_model_instance(
            tenant_id=tenant_id,
            provider=default_model_entity.provider.provider,
            model_type=model_type,
            model=default_model_entity.model,
        )


class LBModelManager:
    def __init__(
        self,
        tenant_id: str,
        provider: str,
        model_type: ModelType,
        model: str,
        load_balancing_configs: list[ModelLoadBalancingConfiguration],
        managed_credentials: Optional[dict] = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._provider = provider
        self._model_type = model_type
        self._model = model
        self._load_balancing_configs = load_balancing_configs

        for config in self._load_balancing_configs[:]:  # Iterate over a copy
            if config.name == "__inherit__":
                if managed_credentials:
                    config.credentials = managed_credentials
                else:
                    self._load_balancing_configs.remove(config)


    def fetch_next(self) -> Optional[ModelLoadBalancingConfiguration]:
        cache_key = (
            f"model_lb_index:{self._tenant_id}:{self._provider}:{self._model_type.value}:{self._model}"
        )
        cooldown_configs = []
        max_index = len(self._load_balancing_configs)

        while True:
            current_index = redis_client.incr(cache_key)
            current_index = cast(int, current_index)
            if current_index >= 10000000:
                current_index = 1
                redis_client.set(cache_key, current_index)

            redis_client.expire(cache_key, 3600)
            current_index %= max_index  # Correct modulo operation

            config = self._load_balancing_configs[current_index]

            if self.in_cooldown(config):
                cooldown_configs.append(config)
                if len(cooldown_configs) == len(self._load_balancing_configs):
                    return None  # All in cooldown
                continue

            if dify_config.DEBUG:
                logger.info(
                    f"LB: id={config.id}, name={config.name}, "
                    f"tenant={self._tenant_id}, provider={self._provider}, "
                    f"type={self._model_type.value}, model={self._model}"
                )
            return config

    def cooldown(self, config: ModelLoadBalancingConfiguration, expire: int = 60) -> None:
        cooldown_key = (
            f"model_lb_index:cooldown:{self._tenant_id}:{self._provider}:"
            f"{self._model_type.value}:{self._model}:{config.id}"
        )
        redis_client.setex(cooldown_key, expire, "true")

    def in_cooldown(self, config: ModelLoadBalancingConfiguration) -> bool:
        cooldown_key = (
            f"model_lb_index:cooldown:{self._tenant_id}:{self._provider}:"
            f"{self._model_type.value}:{self._model}:{config.id}"
        )
        return redis_client.exists(cooldown_key)

    @staticmethod
    def get_config_in_cooldown_and_ttl(
        tenant_id: str, provider: str, model_type: ModelType, model: str, config_id: str
    ) -> tuple[bool, int]:
        cooldown_key = (
            f"model_lb_index:cooldown:{tenant_id}:{provider}:{model_type.value}:{model}:{config_id}"
        )
        ttl = redis_client.ttl(cooldown_key)
        if ttl == -2:
            return False, 0
        return True, ttl

    def get_status(self) -> list[dict]:
        """
        Returns the status of each load balancing configuration.
        """
        status = []
        for config in self._load_balancing_configs:
            in_cooldown, ttl = self.get_config_in_cooldown_and_ttl(
                self._tenant_id, self._provider, self._model_type, self._model, config.id
            )
            status.append({
                "id": config.id,
                "name": config.name,
                "in_cooldown": in_cooldown,
                "cooldown_ttl": ttl,  # 冷却剩余时间 (秒)
                # 在这里添加其他相关的状态信息，例如剩余配额
            })
        return status
