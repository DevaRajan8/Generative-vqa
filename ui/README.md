# VQA Assistant - React Native Mobile App

A beautiful React Native mobile application for Visual Question Answering (VQA) using ensemble AI models with Google authentication.

## Features

- 🔐 **Google OAuth Authentication** - Secure sign-in with Google
- 📸 **Image Selection** - Pick from gallery or capture with camera
- 🤖 **Ensemble VQA** - Automatic routing between base and spatial models
- 🎨 **Beautiful UI** - Modern gradient design with smooth animations
- ⚡ **Real-time Answers** - Fast question answering with model visualization
- 📱 **Cross-Platform** - Works on iOS and Android via Expo Go

## Architecture

### Backend
- **FastAPI** server wrapping the ensemble VQA system
- Two models:
  - **Base Model**: General VQA (39.4% accuracy)
  - **Spatial Model**: Spatial reasoning (28.5% accuracy)
- Automatic question routing based on spatial keywords

### Frontend
- **React Native** with Expo
- **Navigation**: React Navigation
- **State Management**: React Context API
- **UI Components**: Custom components with Material icons
- **Styling**: Custom theme with gradients

## Prerequisites

### Backend Requirements
- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- VQA model checkpoints:
  - `vqa_checkpoint.pt` (base model)
  - `vqa_spatial_checkpoint.pt` (spatial model)

### Frontend Requirements
- Node.js 16+
- npm or yarn
- Expo Go app on your mobile device
- Google Cloud OAuth credentials

## Setup Instructions

### 1. Backend Setup

```bash
# Navigate to project root
cd c:\Users\rdeva\Downloads\vqa_coes

# Install API dependencies
pip install -r requirements_api.txt

# Ensure model checkpoints are in the root directory
# - vqa_checkpoint.pt
# - vqa_spatial_checkpoint.pt

# Start the backend server
python backend_api.py
```

The backend will start on `http://0.0.0.0:8000`

**Important**: Note your computer's local IP address for mobile testing:
- Windows: Run `ipconfig` and find your IPv4 Address
- Mac/Linux: Run `ifconfig` or `ip addr`

### 2. Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **Google+ API**
4. Go to **Credentials** > **Create Credentials** > **OAuth 2.0 Client ID**
5. Create credentials for:
   - **Web application** (for Expo Go)
   - **iOS** (if building standalone iOS app)
   - **Android** (if building standalone Android app)
6. Update `ui/src/config/google.js` with your client IDs

### 3. Frontend Setup

```bash
# Navigate to UI folder
cd ui

# Install dependencies (already done)
npm install

# Update API configuration
# Edit ui/src/config/api.js and replace with your local IP:
# export const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';

# Update Google OAuth configuration
# Edit ui/src/config/google.js with your client IDs
```

### 4. Running the App

```bash
# Make sure backend is running first!

# Start Expo development server
npm start

# Or use specific platform
npm run android  # For Android
npm run ios      # For iOS (Mac only)
```

**Testing on Physical Device:**
1. Install **Expo Go** app from App Store or Play Store
2. Scan the QR code from the terminal
3. Ensure your phone and computer are on the same network
4. The app should load automatically

## Usage

### 1. Sign In
- Open the app
- Tap "Sign in with Google"
- Complete the OAuth flow
- You'll be redirected to the home screen

### 2. Ask Questions
1. **Select Image**:
   - Tap "Camera" to take a photo
   - Tap "Gallery" to choose from library
2. **Enter Question**:
   - Type your question in the text field
   - Examples:
     - "What color is the car?" (uses base model)
     - "What is to the right of the table?" (uses spatial model)
3. **Get Answer**:
   - Tap "Ask Question"
   - View the answer with model type indicator

### 3. Understanding Model Routing

The app automatically routes questions to the appropriate model:

**Spatial Model** (📍) - Used for questions containing:
- Directional: right, left, above, below, top, bottom
- Positional: front, behind, next to, beside, near
- Relational: closest, farthest, nearest

**Base Model** (🔍) - Used for all other questions:
- Object identification
- Color questions
- Counting
- General descriptions

## Project Structure

```
ui/
├── src/
│   ├── config/
│   │   ├── api.js              # API configuration
│   │   └── google.js           # Google OAuth config
│   ├── contexts/
│   │   └── AuthContext.js      # Authentication state
│   ├── screens/
│   │   ├── LoginScreen.js      # Login screen
│   │   └── HomeScreen.js       # Main VQA screen
│   ├── services/
│   │   └── api.js              # API client
│   └── styles/
│       ├── theme.js            # Theme configuration
│       └── globalStyles.js     # Global styles
├── App.js                      # Main app component
├── app.json                    # Expo configuration
└── package.json                # Dependencies
```

## API Endpoints

### Backend API

- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `POST /api/answer` - Answer VQA question
  - Body: `multipart/form-data`
  - Fields: `image` (file), `question` (string)
  - Response: `{ answer, model_used, confidence, question_type }`
- `GET /api/models/info` - Get model information

## Configuration

### API Configuration (`ui/src/config/api.js`)
```javascript
export const API_BASE_URL = 'http://YOUR_LOCAL_IP:8000';
```

### Google OAuth (`ui/src/config/google.js`)
```javascript
export const GOOGLE_CONFIG = {
  webClientId: 'YOUR_WEB_CLIENT_ID.apps.googleusercontent.com',
  iosClientId: 'YOUR_IOS_CLIENT_ID.apps.googleusercontent.com',
  androidClientId: 'YOUR_ANDROID_CLIENT_ID.apps.googleusercontent.com',
};
```

## Troubleshooting

### Cannot Connect to Backend
- Ensure backend server is running (`python backend_api.py`)
- Check that `API_BASE_URL` in `ui/src/config/api.js` matches your local IP
- Verify phone and computer are on the same network
- Check firewall settings

### Google Login Not Working
- Verify OAuth credentials are correctly configured
- Check that redirect URI matches Expo configuration
- Ensure Google+ API is enabled in Cloud Console

### Image Upload Fails
- Check camera/gallery permissions
- Verify image size is reasonable (< 10MB)
- Check backend logs for errors

### Model Loading Issues
- Ensure checkpoint files are in the correct location
- Check GPU/CPU availability
- Verify all Python dependencies are installed

## Building for Production

### Android
```bash
eas build --platform android
```

### iOS
```bash
eas build --platform ios
```

Note: You'll need an Expo account and EAS CLI configured.

## Technologies Used

### Frontend
- React Native
- Expo
- React Navigation
- Expo Auth Session (Google OAuth)
- Expo Image Picker
- Axios
- React Native Paper
- Expo Linear Gradient

### Backend
- FastAPI
- Uvicorn
- PyTorch
- CLIP (OpenAI)
- GPT-2 (Hugging Face)
- Pillow

## Performance

- **Base Model**: 39.4% accuracy on general VQA
- **Spatial Model**: 28.5% accuracy on spatial questions
- **Inference Time**: ~2-5 seconds per question (GPU)
- **Model Size**: ~2GB total (both models)

## License

This project is for educational purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review backend logs
3. Check Expo console for frontend errors

## Credits

- VQA Models: Custom ensemble system
- UI Design: Modern gradient aesthetic
- Icons: Material Community Icons
- Authentication: Google OAuth 2.0
