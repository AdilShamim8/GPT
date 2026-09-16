# Interactive Web Playground Studio

`nano-gpt-prod` comes packaged with a zero-dependency, rich, modern Web Studio for local model interaction, temperature experimentation, and token representation inspection.

## Key Features

1. **Real-time SSE Chat Assistant:**
   - Tokens stream directly to the chat interface using Server-Sent Events (`/v1/chat/completions?stream=true`).
   - Supports multi-turn conversations and role prompts (`user`, `assistant`, `system`).

2. **Interactive Generation Hyperparameters:**
   - **Temperature Slider:** Dynamically tune sampling randomness ($0.05$ to $2.0$).
   - **Top-p Nucleus Slider:** Constrain candidate mass between $10\%$ and $100\%$.
   - **Repetition Penalty:** Eliminate repetitive phrase loops.
   - **Max Token Length:** Limit generation output lengths.

3. **Live Telemetry:**
   - Real-time measurement of generation speed (`tok/s`) and total tokens emitted.

4. **Visual Token Inspector:**
   - Visualizes word and character boundaries with color-coded token pills.
   - Shows byte lengths and character-to-token compression ratios.

## Launching the Web Playground

```bash
# Launch server and automatically open browser at http://localhost:8000
python serve.py --checkpoint checkpoints/shakespeare/ckpt_best.pt --open
```
