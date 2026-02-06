// Global error handler to suppress specific errors
// Add this file to suppress errors that don't affect functionality

import { LogBox } from 'react-native';

// Ignore specific warnings and errors
LogBox.ignoreLogs([
  // Metro bundler cache-related errors
  "Property 'ipconfig' doesn't exist",
  
  // Deprecated API warnings (already fixed in code)
  "ImagePicker.MediaTypeOptions",
  
  // Add more patterns here if needed
]);

// Optionally, ignore all logs (not recommended for development)
// LogBox.ignoreAllLogs();

export default LogBox;
