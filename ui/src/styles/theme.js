import { Platform } from "react-native";

// Theme Configuration - Vibrant & Colorful
export const theme = {
  colors: {
    // Vibrant primary colors
    primary: "#FF6B9D", // Hot Pink
    secondary: "#FEC163", // Golden Yellow
    accent: "#4ECDC4", // Turquoise

    // Background gradients
    background: "#1A1A2E", // Deep Navy
    surface: "#16213E", // Dark Blue
    card: "#0F3460", // Rich Blue

    // Text colors
    text: "#FFFFFF",
    textSecondary: "#E94560", // Coral Red

    // Status colors
    error: "#FF5252",
    success: "#00E676",
    warning: "#FFD600",
    info: "#00B0FF",

    // Gradient colors - Rainbow theme
    gradientStart: "#FF6B9D", // Pink
    gradientMiddle: "#C44569", // Rose
    gradientEnd: "#FEC163", // Gold

    // Alternative gradients
    gradient2Start: "#4ECDC4", // Turquoise
    gradient2Middle: "#556FB5", // Purple Blue
    gradient2End: "#A8E6CF", // Mint

    gradient3Start: "#FFD93D", // Yellow
    gradient3Middle: "#FF6B9D", // Pink
    gradient3End: "#6BCB77", // Green

    // Model badges - Bright colors
    baseModel: "#00D9FF", // Cyan
    spatialModel: "#FFB800", // Amber

    // UI accents
    buttonPrimary: "#FF6B9D",
    buttonSecondary: "#4ECDC4",
    inputBg: "#0F3460",
    inputBorder: "#FF6B9D",
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
        shadowColor: "#FF6B9D",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 4,
      },
      android: {
        elevation: 3,
      },
      default: {
        elevation: 3,
      },
    }),
    md: Platform.select({
      ios: {
        shadowColor: "#FF6B9D",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
      },
      android: {
        elevation: 5,
      },
      default: {
        elevation: 5,
      },
    }),
    lg: Platform.select({
      ios: {
        shadowColor: "#FF6B9D",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.4,
        shadowRadius: 16,
      },
      android: {
        elevation: 8,
      },
      default: {
        elevation: 8,
      },
    }),
  },
};
