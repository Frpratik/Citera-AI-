// Citera AI web page: plain JavaScript, no build step, no libraries.
//
// Safety rules this file follows (tests/test_frontend.py checks them):
//  * Everything that comes from the server (answers, file names, snippets, errors) is put on the
//    page with textContent / createElement, NEVER as HTML. Text from a PDF or from the AI can
//    therefore never run as code.
//  * The login token is kept in sessionStorage (gone when the tab closes), not localStorage.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ small helpers
  function store(action, key, value) {
    try {
      if (action === "get") return sessionStorage.getItem(key);
      if (action === "set") sessionStorage.setItem(key, value);
      if (action === "remove") sessionStorage.removeItem(key);
    } catch (e) { /* storage blocked: the app still works, it just forgets on reload */ }
    return null;
  }

  // Builds an element. Text always goes in through textContent.
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of children) if (child) node.append(child);
    return node;
  }

  function errorMessage(data, status) {
    const details = data && data.error && data.error.details;
    if (details && details.length) return details.map((d) => d.message).join("; ");
    return (data && data.error && data.error.message) || `Request failed (${status})`;
  }

  const ANSWER_LABELS = {
    document_grounded: "From your documents",
    insufficient_evidence: "Not found in your documents",
    conflicting_information: "Your documents disagree",
    general: "General answer",
    passages_only: "AI unavailable: matching passages",
  };

  // ------------------------------------------------------------------ state
  const state = {
    token: store("get", "citera_token"),
    email: store("get", "citera_email"),
    mode: "login",
    pollTimer: null,
    chatBusy: false,
    processingNotice: false,
  };

  // ------------------------------------------------------------------ talking to the server
  async function api(path, { method = "GET", body, form } = {}) {
    const headers = {};
    if (state.token) headers.Authorization = `Bearer ${state.token}`;
    let payload;
    if (form) {
      payload = form;
    } else if (body !== undefined) {
      headers["Content-Type"] = "application/json";
      payload = JSON.stringify(body);
    }
    const response = await fetch(path, { method, headers, body: payload });
    let data = null;
    try { data = await response.json(); } catch (e) { /* an empty reply */ }
    if (!response.ok) {
      if (response.status === 401 && state.token) logout("Your session ended. Please log in again.");
      throw new Error(errorMessage(data, response.status));
    }
    return data;
  }

  // ------------------------------------------------------------------ sign in / out
  function setMode(mode) {
    state.mode = mode;
    const registering = mode === "register";
    $("tab-login").classList.toggle("active", !registering);
    $("tab-register").classList.toggle("active", registering);
    $("tab-login").setAttribute("aria-selected", String(!registering));
    $("tab-register").setAttribute("aria-selected", String(registering));
    $("auth-submit").textContent = registering ? "Create account" : "Log in";
    $("password").setAttribute("autocomplete", registering ? "new-password" : "current-password");
    $("password-hint").hidden = !registering;
    $("auth-error").textContent = "";
  }

  async function submitAuth(event) {
    event.preventDefault();
    const email = $("email").value.trim();
    const password = $("password").value;
    $("auth-error").textContent = "";
    $("auth-submit").disabled = true;
    try {
      if (state.mode === "register") {
        await api("/api/auth/register", { method: "POST", body: { email, password } });
      }
      const result = await api("/api/auth/login", { method: "POST", body: { email, password } });
      state.token = result.access_token;
      state.email = email.toLowerCase();
      store("set", "citera_token", state.token);
      store("set", "citera_email", state.email);
      $("password").value = "";
      await enterApp();
    } catch (e) {
      $("auth-error").textContent = e.message;
    } finally {
      $("auth-submit").disabled = false;
    }
  }

  async function enterApp() {
    $("auth-view").hidden = true;
    $("app-view").hidden = false;
    $("user-area").hidden = false;
    $("user-email").textContent = state.email || "";
    await Promise.all([loadDocuments(), loadHistory()]);
    $("message-input").focus();
  }

  function logout(message = "") {
    state.token = null;
    state.email = null;
    store("remove", "citera_token");
    store("remove", "citera_email");
    stopPolling();
    $("doc-list").replaceChildren();
    $("messages").replaceChildren();
    $("app-view").hidden = true;
    $("user-area").hidden = true;
    $("auth-view").hidden = false;
    $("auth-error").textContent = message;
  }

  // ------------------------------------------------------------------ documents
  function renderDocuments(documents) {
    const list = $("doc-list");
    list.replaceChildren();
    $("doc-empty").hidden = documents.length > 0;
    for (const doc of documents) {
      const meta = doc.status === "ready" ? `${doc.page_count} pages, ${doc.chunk_count} chunks` : "";
      const actions = el("div", { class: "doc-actions" });
      if (doc.status === "failed") {
        const retry = el("button", { type: "button", class: "link-btn", text: "Try again" });
        retry.addEventListener("click", () => reprocessDocument(doc.id));
        actions.append(retry);
      }
      const remove = el("button", { type: "button", class: "link-btn danger", text: "Delete" });
      remove.addEventListener("click", () => deleteDocument(doc));
      actions.append(remove);

      list.append(el("li", { class: "doc" },
        el("div", { class: "doc-name", text: doc.filename }),
        el("div", { class: "doc-meta" }, el("span", { class: `badge ${doc.status}`, text: doc.status }), meta ? ` ${meta}` : ""),
        doc.error ? el("div", { class: "doc-error", text: doc.error }) : null,
        actions));
    }
  }

  function stopPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = null;
  }

  async function loadDocuments() {
    try {
      const { documents } = await api("/api/documents");
      renderDocuments(documents);
      // Documents are processed in the background: keep checking while any is still working.
      const working = documents.some((d) => d.status === "processing");
      if (working && !state.pollTimer) state.pollTimer = setInterval(loadDocuments, 2000);
      if (!working) {
        stopPolling();
        if (state.processingNotice) {  // the "being processed" note is no longer true
          $("upload-status").textContent = "";
          state.processingNotice = false;
        }
      }
    } catch (e) {
      stopPolling();
    }
  }

  async function uploadFiles(files) {
    for (const file of files) {
      $("upload-status").textContent = `Uploading ${file.name}...`;
      const form = new FormData();
      form.append("file", file);
      try {
        await api("/api/documents", { method: "POST", form });
        $("upload-status").textContent = `${file.name} uploaded. It is being processed.`;
        state.processingNotice = true;  // cleared by loadDocuments once nothing is processing any more
      } catch (e) {
        $("upload-status").textContent = `${file.name}: ${e.message}`;
      }
      await loadDocuments();
    }
    $("file-input").value = "";
  }

  async function deleteDocument(doc) {
    if (!window.confirm(`Delete "${doc.filename}"? This cannot be undone.`)) return;
    try { await api(`/api/documents/${doc.id}`, { method: "DELETE" }); } catch (e) { $("upload-status").textContent = e.message; }
    await loadDocuments();
  }

  async function reprocessDocument(id) {
    try { await api(`/api/documents/${id}/reprocess`, { method: "POST" }); } catch (e) { $("upload-status").textContent = e.message; }
    await loadDocuments();
  }

  // ------------------------------------------------------------------ chat
  function scrollToEnd() {
    const box = $("messages");
    box.scrollTop = box.scrollHeight;
  }

  function addMessage(role, text) {
    const empty = $("messages").querySelector(".empty-chat");
    if (empty) empty.remove();
    const body = el("div", { class: "msg-text", text });
    const node = el("div", { class: `msg ${role}` }, body);
    $("messages").append(node);
    scrollToEnd();
    return { node, body };
  }

  function renderAnswer(message, result) {
    message.body.textContent = result.answer;
    const meta = el("div", { class: "msg-meta" },
      el("span", { class: `badge ${result.answer_type}`, text: ANSWER_LABELS[result.answer_type] || result.answer_type }));
    if (result.evidence_status === "strong" || result.evidence_status === "moderate") {
      meta.append(el("span", { class: `badge ${result.evidence_status}`, text: `${result.evidence_status} evidence` }));
    }
    message.node.append(meta);
    for (const warning of result.warnings || []) message.node.append(el("div", { class: "notice", text: warning }));
    if (result.sources && result.sources.length) {
      const items = el("ul");
      for (const s of result.sources) {
        items.append(el("li", {},
          el("strong", { text: `${s.document_name}, page ${s.page}` }),
          el("blockquote", { text: s.snippet })));
      }
      message.node.append(el("details", { class: "sources" }, el("summary", { text: `Sources (${result.sources.length})` }), items));
    }
    scrollToEnd();
  }

  function showMessageError(message, text) {
    message.body.className = "msg-text msg-error";
    message.body.textContent = text;
  }

  // Reads the server's event stream: blocks of "event: name" + "data: {json}" split by blank lines.
  function parseEvent(block) {
    let name = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7);
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    return { name, data: data ? JSON.parse(data) : {} };
  }

  async function sendMessage(text) {
    state.chatBusy = true;
    $("send-btn").disabled = true;
    addMessage("user", text);
    const reply = addMessage("assistant", "Thinking...");
    reply.body.className = "msg-text msg-status";
    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.token}` },
        body: JSON.stringify({ message: text }),
      });
      if (!response.ok) {
        let data = null;
        try { data = await response.json(); } catch (e) { /* no body */ }
        if (response.status === 401) logout("Your session ended. Please log in again.");
        throw new Error(errorMessage(data, response.status));
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let end;
        while ((end = buffer.indexOf("\n\n")) !== -1) {
          const event = parseEvent(buffer.slice(0, end));
          buffer = buffer.slice(end + 2);
          if (event.name === "step") reply.body.textContent = `${event.data.label || event.data.step}...`;
          else if (event.name === "answer") { reply.body.className = "msg-text"; renderAnswer(reply, event.data); }
          else if (event.name === "error") showMessageError(reply, event.data.message || "Something went wrong.");
        }
      }
    } catch (e) {
      showMessageError(reply, e.message);
    } finally {
      state.chatBusy = false;
      $("send-btn").disabled = false;
      $("message-input").focus();
    }
  }

  async function loadHistory() {
    const box = $("messages");
    box.replaceChildren();
    try {
      const { messages } = await api("/api/chat/history");
      for (const m of messages) {
        const shown = addMessage(m.role, m.content);
        if (m.role === "assistant" && m.answer_type) {
          shown.node.append(el("div", { class: "msg-meta" },
            el("span", { class: `badge ${m.answer_type}`, text: ANSWER_LABELS[m.answer_type] || m.answer_type })));
        }
      }
    } catch (e) { /* leave the chat empty */ }
    if (!box.children.length) {
      box.append(el("p", { class: "empty-chat", text: "Ask a question about your documents. Answers show the page they came from." }));
    }
    scrollToEnd();
  }

  async function clearHistory() {
    if (!window.confirm("Clear your chat history?")) return;
    try { await api("/api/chat/history", { method: "DELETE" }); } catch (e) { /* ignore */ }
    await loadHistory();
  }

  // ------------------------------------------------------------------ usage
  async function showUsage() {
    const list = $("usage-list");
    list.replaceChildren();
    try {
      const u = await api("/api/usage");
      const rows = [
        ["Questions", u.questions], ["AI calls", u.llm_calls], ["Tokens in", u.input_tokens],
        ["Tokens out", u.output_tokens], ["Average answer time", `${(u.avg_response_ms / 1000).toFixed(1)} s`],
        ["Answers using the backup AI", u.fallbacks],
      ];
      if (u.estimated_cost !== null) rows.push(["Estimated cost", u.estimated_cost.toFixed(4)]);
      for (const [label, value] of rows) list.append(el("dt", { text: label }), el("dd", { text: String(value) }));
    } catch (e) {
      list.append(el("dt", { text: "Error" }), el("dd", { text: e.message }));
    }
    $("usage-dialog").showModal();
  }

  // ------------------------------------------------------------------ wiring
  $("tab-login").addEventListener("click", () => setMode("login"));
  $("tab-register").addEventListener("click", () => setMode("register"));
  $("auth-form").addEventListener("submit", submitAuth);
  $("logout-btn").addEventListener("click", () => logout());
  $("usage-btn").addEventListener("click", showUsage);
  $("usage-close").addEventListener("click", () => $("usage-dialog").close());
  $("clear-btn").addEventListener("click", clearHistory);
  $("file-input").addEventListener("change", (e) => uploadFiles(Array.from(e.target.files)));

  $("chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const text = $("message-input").value.trim();
    if (!text || state.chatBusy) return;
    $("message-input").value = "";
    sendMessage(text);
  });
  $("message-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {  // Enter sends; Shift+Enter makes a new line
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });

  // On page load: if a token survived in this tab, check it is still good.
  if (state.token) {
    api("/api/auth/me").then((me) => { state.email = me.email; return enterApp(); }).catch(() => logout());
  }
})();
