document.addEventListener("DOMContentLoaded", () => {
  const inspectorInput = document.getElementById("inspectorInput");
  const tokenPills = document.getElementById("tokenPills");
  const tokenSummary = document.getElementById("tokenSummary");

  const colors = [
    "rgba(99, 102, 241, 0.25)",
    "rgba(168, 85, 247, 0.25)",
    "rgba(236, 72, 153, 0.25)",
    "rgba(34, 197, 94, 0.25)",
    "rgba(234, 179, 8, 0.25)",
    "rgba(14, 165, 233, 0.25)",
  ];

  function updateInspector() {
    const text = inspectorInput.value;
    tokenPills.innerHTML = "";

    if (!text) {
      tokenSummary.textContent = "0 tokens | 0 characters";
      return;
    }

    // Tokenize text into words/chunks for client inspection demonstration
    // Matches whitespace or non-whitespace sequences
    const tokens = text.match(/\s+|\S+/g) || [];

    tokens.forEach((tok, i) => {
      const pill = document.createElement("div");
      pill.className = "token-pill";
      pill.style.backgroundColor = colors[i % colors.length];

      const textSpan = document.createElement("span");
      textSpan.textContent = tok.replace(/ /g, "·").replace(/\n/g, "↵\n");

      const idSpan = document.createElement("span");
      idSpan.className = "token-pill-id";
      idSpan.textContent = `#${i}`;

      pill.appendChild(textSpan);
      pill.appendChild(idSpan);
      tokenPills.appendChild(pill);
    });

    const byteLen = new TextEncoder().encode(text).length;
    tokenSummary.innerHTML = `
      <span><strong>Total Tokens:</strong> ${tokens.length}</span> &bull;
      <span><strong>Characters:</strong> ${text.length}</span> &bull;
      <span><strong>Bytes:</strong> ${byteLen}</span> &bull;
      <span><strong>Compression Ratio:</strong> ${(text.length / Math.max(tokens.length, 1)).toFixed(2)} chars/token</span>
    `;
  }

  if (inspectorInput) {
    inspectorInput.addEventListener("input", updateInspector);
    updateInspector();
  }
});
