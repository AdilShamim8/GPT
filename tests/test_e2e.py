import tempfile
from pathlib import Path
import torch
from gpt.config import ModelConfig, TrainingConfig
from gpt.data import InMemoryDataset
from gpt.inference import TextGenerator
from gpt.model.gpt import GPT
from gpt.serve.chat import handle_chat_completion
from gpt.serve.types import ChatCompletionRequest
from gpt.tokenizer.char_tokenizer import CharTokenizer
from gpt.training import CheckpointManager, Trainer


def test_end_to_end_pipeline():
    """Verify complete lifecycle: tokenize -> train -> checkpoint -> load -> infer -> serve."""
    # 1. Dataset & Tokenizer with special tokens enabled (<|unk|> fallback)
    base_text = "First Citizen:\nBefore we proceed any further, hear me speak.\nAll:\nSpeak, speak.\n"
    corpus = base_text * 8
    tokenizer = CharTokenizer.from_text(corpus, add_special_tokens=True)

    # 2. Configs
    model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        train_cfg = TrainingConfig(
            max_iters=5,
            batch_size=2,
            checkpoint_dir=tmpdir,
            eval_interval=5,
            eval_iters=1,
            log_interval=2,
            device="cpu",
        )

        train_ds, val_ds = InMemoryDataset.from_text(
            text=corpus,
            tokenizer=tokenizer,
            block_size=model_cfg.block_size,
            split_ratio=0.8,
        )

        # 3. Train
        model = GPT(model_cfg)
        trainer = Trainer(model=model, config=train_cfg, train_dataset=train_ds, val_dataset=val_ds)
        trainer.train()

        # 4. Save checkpoint
        ckpt_mgr = CheckpointManager(tmpdir)
        ckpt_path = ckpt_mgr.save(model, trainer.optimizer, step=5, val_loss=2.1)
        assert ckpt_path.exists()

        # 5. Load checkpoint
        loaded_model, loaded_step, _ = ckpt_mgr.load_latest(device="cpu")
        assert loaded_step == 5

        # 6. Generate text
        gen = TextGenerator(loaded_model, tokenizer, device="cpu")
        res = gen.generate(prompt="First", max_new_tokens=10)
        assert len(res.text) > len("First")

        # 7. OpenAI API format verification
        chat_req = ChatCompletionRequest(
            messages=[{"role": "user", "content": "Speak"}],
            max_tokens=10,
        )
        chat_resp = handle_chat_completion(gen, chat_req)
        assert "choices" in chat_resp
        assert len(chat_resp["choices"]) > 0
        assert "content" in chat_resp["choices"][0]["message"]


if __name__ == "__main__":
    test_end_to_end_pipeline()
    print("END-TO-END PIPELINE INTEGRATION TEST PASSED!")
