import axios from 'axios';
import { Platform } from 'react-native';
import { API_BASE_URL, API_ENDPOINTS, API_TIMEOUT } from '../config/api';
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
});
apiClient.interceptors.request.use(
  (config) => {
    console.log('API Request:', config.method?.toUpperCase(), config.url);
    return config;
  },
  (error) => {
    console.error('Request Error:', error);
    return Promise.reject(error);
  }
);
apiClient.interceptors.response.use(
  (response) => {
    console.log('API Response:', response.status, response.config.url);
    return response;
  },
  (error) => {
    console.error('Response Error:', error.response?.status, error.message);
    return Promise.reject(error);
  }
);
/**
 * On web, expo-image-picker returns a blob: URL.
 * Browsers require an actual Blob/File in FormData — the React Native
 * { uri, name, type } shorthand only works in the native runtime.
 * We fetch the blob on web and fall back to the native object on iOS/Android.
 */
const appendImageToFormData = async (formData, imageUri) => {
  if (Platform.OS === 'web') {
    // Fetch the blob: or data: URI and convert to a File object
    const response = await fetch(imageUri);
    const blob = await response.blob();
    const filename = imageUri.split('/').pop() || 'image.jpg';
    const ext = filename.split('.').pop() || 'jpg';
    const mimeType = blob.type || `image/${ext}`;
    const file = new File([blob], filename || `photo.${ext}`, { type: mimeType });
    formData.append('image', file);
  } else {
    // React Native native runtime supports the { uri, name, type } shorthand
    const filename = imageUri.split('/').pop() || 'photo.jpg';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : 'image/jpeg';
    formData.append('image', { uri: imageUri, name: filename, type });
  }
};
export const askQuestion = async (imageUri, question) => {
  try {
    const formData = new FormData();
    await appendImageToFormData(formData, imageUri);
    formData.append('question', question);
    // On web: do NOT set Content-Type manually — the browser adds the boundary automatically.
    // On native: React Native's XHR needs the hint.
    const headers = Platform.OS !== 'web' ? { 'Content-Type': 'multipart/form-data' } : {};
    const response = await apiClient.post(API_ENDPOINTS.ANSWER, formData, { headers });
    return response.data;
  } catch (error) {
    console.error('askQuestion error:', error);
    if (error.response) {
      throw new Error(error.response.data.detail || 'Server error');
    } else if (error.request) {
      throw new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      throw new Error(error.message || 'An error occurred');
    }
  }
};
export const checkHealth = async () => {
  try {
    const response = await apiClient.get(API_ENDPOINTS.HEALTH);
    return response.data;
  } catch (error) {
    console.error('checkHealth error:', error);
    throw error;
  }
};
export const getModelsInfo = async () => {
  try {
    const response = await apiClient.get(API_ENDPOINTS.MODELS_INFO);
    return response.data;
  } catch (error) {
    console.error('getModelsInfo error:', error);
    throw error;
  }
};
export default {
  askQuestion,
  checkHealth,
  getModelsInfo,
};