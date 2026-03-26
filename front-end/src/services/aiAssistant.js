import axios from "axios";

const aiApi = axios.create({
  baseURL: "http://localhost:8001",
  timeout: 120000,
});

export async function sendChatMessage(message, sessionId = "default") {
  const response = await aiApi.post("/chat", {
    message,
    session_id: sessionId,
  });
  return response.data;
}

export async function getChatHistory(sessionId) {
  const response = await aiApi.get(`/chat/history/${sessionId}`);
  return response.data;
}

export async function clearChatHistory(sessionId) {
  const response = await aiApi.delete(`/chat/history/${sessionId}`);
  return response.data;
}

export async function checkHealth() {
  const response = await aiApi.get("/health");
  return response.data;
}
