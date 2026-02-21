import { Platform } from "react-native";
export const theme = {
  colors: {
    primary: "#2563EB", 
    secondary: "#1E40AF", 
    accent: "#3B82F6", 
    background: "#0F172A", 
    surface: "#1E293B", 
    card: "#1E293B", 
    text: "#FFFFFF", 
    textSecondary: "#94A3B8", 
    error: "#EF4444", 
    success: "#10B981", 
    warning: "#F59E0B", 
    info: "#3B82F6", 
    gradientStart: "#1E40AF", 
    gradientMiddle: "#2563EB", 
    gradientEnd: "#3B82F6", 
    gradient2Start: "#0F172A", 
    gradient2Middle: "#1E293B", 
    gradient2End: "#334155", 
    baseModel: "#3B82F6", 
    spatialModel: "#2563EB", 
    buttonPrimary: "#2563EB",
    buttonSecondary: "#1E293B",
    inputBg: "#1E293B",
    inputBorder: "#334155", 
    glassBackground: "rgba(30, 41, 59, 0.7)", 
    glassBorder: "rgba(59, 130, 246, 0.3)", 
  },
  animations: {
    typingSpeed: 30, 
    skeletonPulse: 1500, 
    confidenceAnimation: 1000, 
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
    xxl: 48,
  },
  borderRadius: {
    sm: 8,
    md: 12,
    lg: 16,
    xl: 24,
    full: 9999,
  },
  typography: {
    h1: {
      fontSize: 32,
      fontWeight: "bold",
      lineHeight: 40,
    },
    h2: {
      fontSize: 24,
      fontWeight: "bold",
      lineHeight: 32,
    },
    h3: {
      fontSize: 20,
      fontWeight: "600",
      lineHeight: 28,
    },
    body: {
      fontSize: 16,
      fontWeight: "normal",
      lineHeight: 24,
    },
    caption: {
      fontSize: 14,
      fontWeight: "normal",
      lineHeight: 20,
    },
    small: {
      fontSize: 12,
      fontWeight: "normal",
      lineHeight: 16,
    },
  },
  shadows: {
    sm: Platform.select({
      ios: {
        shadowColor: "#000000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
      },
      android: {
        elevation: 2,
      },
      default: {
        elevation: 2,
      },
    }),
    md: Platform.select({
      ios: {
        shadowColor: "#000000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 8,
      },
      android: {
        elevation: 4,
      },
      default: {
        elevation: 4,
      },
    }),
    lg: Platform.select({
      ios: {
        shadowColor: "#000000",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.2,
        shadowRadius: 16,
      },
      android: {
        elevation: 6,
      },
      default: {
        elevation: 6,
      },
    }),
  },
};