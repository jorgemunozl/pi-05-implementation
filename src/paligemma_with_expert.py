import torch.nn as nn
from typing import Literal


class PaliGemmaWithExpertModel(nn.Module):
    """PaliGemma model with action expert for PI05."""
    def __init__(
        self, vlm_config, action_expert_config, use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        # Configuration from the VLM PALIGEMMA
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = (
            vlm_config.width if use_adarms[0] else None
        )
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        # CONFIGURATION FOR THE ACTION EXPERT
        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=(
                action_expert_config.width if use_adarms[1] else None
            ),
        )

        self.paligemma = PaliGemmaForConditionalGeneration(
            config=vlm_config_hf)

        # Expert Architecture Initialized from a small Gemma Version
        # From pretrained only loads the paligemma model
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(
            self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    @torch.no_grad()
    def sample_low_level_task(
        self,                              # HERE IS KEY
        pixel_values: torch.Tensor,       # [B, C, H, W]
        input_ids: torch.Tensor,          # [B, L] - The tokenized prompt
        max_decoding_steps: int = 20,
        eos_token_id: int = 1,
        temperature: float = 0.0,
    ):
        device = pixel_values.device
        batch_size = pixel_values.shape[0]

        # 1. PROCESS VISION AND PROMPT (The "Context")
        # Get vision features and combine with text embeddings
        # Most HF PaliGemma models do this in their 'get_input_embeddings' or forward logic
        inputs_embeds = self.paligemma.get_input_embeddings()(input_ids)
        vision_outputs = self.paligemma.vision_tower(pixel_values.to(self.paligemma.dtype))
        selected_image_feature = vision_outputs.last_hidden_state
        image_features = self.paligemma.multi_modal_projector(selected_image_feature)

        # Merge Image [B, 256, D] and Text [B, L, D]
        # Note: PaliGemma usually expects <image_tokens>...<text_tokens>
        # We assume input_ids already contains the placeholder for images
        combined_embeds = self._merge_embeddings(image_features, inputs_embeds)

        # 2. INITIAL PREFILL (First pass to get KV Cache)
        outputs = self.paligemma.language_model(
            inputs_embeds=combined_embeds,
            use_cache=True,
            return_dict=True
        )

        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]  # [B, Vocab]

        output_tokens = torch.zeros((batch_size, max_decoding_steps),
                                    dtype=torch.long, device=device)
        all_eos = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 3. AUTOREGRESSIVE LOOP (Thinking)
        for step in range(max_decoding_steps):
            # A. Sample next token
            if temperature > 0.0:
                probs = F.softmax(last_logits / temperature, dim=-1)
                token = torch.multinomial(probs, num_samples=1)
            else:
                token = torch.argmax(last_logits, dim=-1, keepdim=True)

            output_tokens[:, step] = token.squeeze(-1)

            # Check for EOS
            all_eos |= (token.squeeze(-1) == eos_token_id)
            if all_eos.all():
                break

            # B. Feed the new token back in
            next_token_embeds = self.paligemma.get_input_embeddings()(token)
            outputs = self.paligemma.language_model(
                inputs_embeds=next_token_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

        # We return the tokens and the KV cache
        # (the 'expert' needs this cache later)
        return output_tokens, past_key_values

    def _merge_embeddings(self, image_features, text_embeds):
        # Specific logic for PaliGemma: image tokens usually come first.
        # This replaces the special <image> tokens in your
        # text_embeds with image_features.
        # Simplified:
        return torch.cat([image_features, text_embeds], dim=1)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]

        # Interesting how it works.
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=(
                    adarms_cond[0] if adarms_cond is not None else None
                )
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None

        # Because outputs_embeds call with inputs_embeds=[None, suffix_embs]
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=(
                    adarms_cond[1] if adarms_cond is not None else None
                )
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (
                hasattr(self, "gradient_checkpointing")
                and self.gradient_checkpointing
                and self.training
            )

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        paligemma=self.paligemma,
                        gemma_expert=self.gemma_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states,
                                                cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds,
                                                     adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        # You only care about the suffix_output to denoise actions.
        return [prefix_output, suffix_output], prefix_past_key_values
