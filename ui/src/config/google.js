// Google OAuth Configuration
// You need to create OAuth credentials in Google Cloud Console
// https://console.cloud.google.com/

export const GOOGLE_CONFIG = {
  // Web Client ID from Google Cloud Console
  webClientId: 'YOUR_WEB_CLIENT_ID.apps.googleusercontent.com',
  
  // iOS Client ID (if building for iOS)
  iosClientId: 'YOUR_IOS_CLIENT_ID.apps.googleusercontent.com',
  
  // Android Client ID (if building for Android)
  androidClientId: 'YOUR_ANDROID_CLIENT_ID.apps.googleusercontent.com',
  
  // Scopes
  scopes: ['profile', 'email'],
  
  // Redirect URI
  redirectUri: 'https://auth.expo.io/@your-username/ui',
};

// Instructions:
// 1. Go to https://console.cloud.google.com/
// 2. Create a new project or select existing
// 3. Enable Google+ API
// 4. Go to Credentials > Create Credentials > OAuth 2.0 Client ID
// 5. Create credentials for:
//    - Web application (for Expo Go)
//    - iOS (if building standalone iOS app)
//    - Android (if building standalone Android app)
// 6. Replace the placeholder values above with your actual client IDs
