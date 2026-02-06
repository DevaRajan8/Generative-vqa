// API Configuration
// Update API_BASE_URL with your local IP address for testing on physical device

// For local testing (emulator/simulator)
// export const API_BASE_URL = 'http://localhost:8000';

// For physical device testing (replace with your computer's local IP)
// Find your IP: Run 'ipconfig' on Windows or 'ifconfig' on Mac/Linux
// export const API_BASE_URL = "http://10.215.4.143:8000";

// Using ngrok for public access (recommended for mobile)
export const API_BASE_URL = "https://key-content-rodent.ngrok-free.app";

export const API_ENDPOINTS = {
  HEALTH: "/health",
  ANSWER: "/api/answer",
  MODELS_INFO: "/api/models/info",
};

export const API_TIMEOUT = 30000; // 30 seconds
