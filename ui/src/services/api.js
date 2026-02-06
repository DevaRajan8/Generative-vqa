import axios from 'axios';
import { API_BASE_URL, API_ENDPOINTS, API_TIMEOUT } from '../config/api';

// Create axios instance
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: API_TIMEOUT,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

// Request interceptor
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

// Response interceptor
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
 * Ask a question about an image
 * @param {string} imageUri - Local URI of the image
 * @param {string} question - Question text
 * @returns {Promise} - API response with answer
 */
export const askQuestion = async (imageUri, question) => {
  try {
    // Create form data
    const formData = new FormData();
    
    // Add image
    const filename = imageUri.split('/').pop();
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : 'image/jpeg';
    
    formData.append('image', {
      uri: imageUri,
      name: filename,
      type: type,
    });
    
    // Add question
    formData.append('question', question);
    
    // Make request
    const response = await apiClient.post(API_ENDPOINTS.ANSWER, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  } catch (error) {
    console.error('askQuestion error:', error);
    
    if (error.response) {
      // Server responded with error
      throw new Error(error.response.data.detail || 'Server error');
    } else if (error.request) {
      // Request made but no response
      throw new Error('Cannot connect to server. Please check if the backend is running.');
    } else {
      // Something else happened
      throw new Error(error.message || 'An error occurred');
    }
  }
};

/**
 * Check API health
 * @returns {Promise} - Health status
 */
export const checkHealth = async () => {
  try {
    const response = await apiClient.get(API_ENDPOINTS.HEALTH);
    return response.data;
  } catch (error) {
    console.error('checkHealth error:', error);
    throw error;
  }
};

/**
 * Get models information
 * @returns {Promise} - Models info
 */
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
