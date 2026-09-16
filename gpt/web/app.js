document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const promptInput = document.getElementById("promptInput");
  const sendBtn = document.getElementById("sendBtn");
  const chatMessages = document.getElementById("chatMessages");
  const clearChatBtn = document.getElementById("clearChatBtn");

  // Sliders
  const tempSlider = document.getElementById("tempSlider");
  const tempValue = document.getElementById("tempValue");
  const topPSlider = document.getElementById("topPSlider");
  const topPValue = document.getElementById("topPValue");
  const maxTokensSlider = document.getElementById("maxTokensSlider");
  const maxTokensValue = document.getElementById("maxTokensValue");
  const repPenaltySlider = document.getElementById("repPenaltySlider");
  const repPenaltyValue = document.getElementById("repPenaltyValue");

  // Telemetry
  const statSpeed = document.getElementById("statSpeed");
  const statTokens = document.getElementById("statTokens");

  // Tabs
  const tabChat = document.getElementById("tabChat");
  const tabInspector = document.getElementById("tabInspector");
  const chatView = document.getElementById("chatView");
  const inspectorView = document.getElementById("inspectorView");

  // Slider bindings
  tempSlider.addEventListener("input", (e) => (tempValue.textContent = parseFloat(e.target.value).toFixed(2)));
  topPSlider.addEventListener("input", (e) => (topPValue.textContent = parseFloat(e.target.value).toFixed(2)));
  maxTokensSlider.addEventListener("input", (e) => (maxTokensValue.textContent = e.target.value));
  repPenaltySlider.addEventListener("input", (e) => (repPenaltyValue.textContent = parseFloat(e.target.value).toFixed(2)));

  // Tab switching
  tabChat.addEventListener("click", () => {
    tabChat.classList.add("active");
    tabInspector.classList.remove("active");
    chatView.classList.add("active");
    inspectorView.classList.remove("active");
  });

  tabInspector.addEventListener("click", () => {
    tabInspector.classList.add("active");
    tabChat.classList.remove("active");
    inspectorView.classList.add("active");
    chatView.classList.remove("active");
  });

  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="message-bubble system-welcome">
        <div class="bubble-header">System</div>
        <div class="bubble-content">Context reset. Ready for new conversation.</div>
      </div>
    `;
  });

  // Conversation history
  let messages = [];

  function appendMessage(role, content) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;
    const header = document.createElement("div");
    header.className = "bubble-header";
    header.textContent = role === "user" ? "You" : "nano-gpt";
    const body = document.createElement("div");
    body.className = "bubble-content";
    body.textContent = content;

    bubble.appendChild(header);
    bubble.appendChild(body);
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return body;
  }

  async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text) return;

    appendMessage("user", text);
    messages.push({ role: "user", content: text });
    promptInput.value = "";

    const assistantContentElem = appendMessage("assistant", "");
    const startTime = performance.now();
    let tokenCount = 0;

    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages,
          temperature: parseFloat(tempSlider.value),
          top_p: parseFloat(topPSlider.value),
          max_tokens: parseInt(maxTokensSlider.value),
          repetition_penalty: parseFloat(repPenaltySlider.value),
          stream: true,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let fullAssistantReply = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split("\n");

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6).trim();
            if (dataStr === "[DONE]") break;
            try {
              const parsed = JSON.parse(dataStr);
              const delta = parsed.choices[0]?.delta?.content;
              if (delta) {
                fullAssistantReply += delta;
                assistantContentElem.textContent = fullAssistantReply;
                chatMessages.scrollTop = chatMessages.scrollHeight;
                tokenCount++;
              }
            } catch (err) {}
          }
        }
      }

      messages.push({ role: "assistant", content: fullAssistantReply });

      const elapsedSec = (performance.now() - startTime) / 1000.0;
      const speed = elapsedSec > 0 ? (tokenCount / elapsedSec).toFixed(1) : 0;
      statSpeed.textContent = `${speed} tok/s`;
      statTokens.textContent = tokenCount;
    } catch (err) {
      assistantContentElem.textContent = "Error communicating with server: " + err.message;
    }
  }

  sendBtn.addEventListener("click", sendMessage);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
});
