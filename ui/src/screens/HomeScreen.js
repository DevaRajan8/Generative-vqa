import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from "react-native";
import { Image } from "react-native";
import { LinearGradient } from "expo-linear-gradient";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import { useAuth } from "../contexts/AuthContext";
import { askQuestion } from "../services/api";
import { theme } from "../styles/theme";

export default function HomeScreen() {
  const { user, signOut } = useAuth();
  const [selectedImage, setSelectedImage] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);

  const pickImage = async (useCamera = false) => {
    try {
      // Request permissions
      const permissionResult = useCamera
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();

      if (!permissionResult.granted) {
        Alert.alert(
          "Permission Required",
          "Please grant camera/gallery permissions to continue.",
        );
        return;
      }

      // Launch picker
      const result = useCamera
        ? await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
          })
        : await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
          });

      if (!result.canceled) {
        const imageUri = result.assets[0].uri;
        console.log("Image selected:", imageUri);
        console.log("Image details:", result.assets[0]);
        setSelectedImage(imageUri);
        setAnswer(null); // Clear previous answer
      }
    } catch (error) {
      console.error("Error picking image:", error);
      Alert.alert("Error", "Failed to pick image");
    }
  };

  const handleAskQuestion = async () => {
    if (!selectedImage) {
      Alert.alert("No Image", "Please select an image first");
      return;
    }

    if (!question.trim()) {
      Alert.alert("No Question", "Please enter a question");
      return;
    }

    try {
      setLoading(true);
      const result = await askQuestion(selectedImage, question);
      setAnswer(result);
    } catch (error) {
      console.error("Error asking question:", error);
      Alert.alert("Error", error.message || "Failed to get answer");
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setSelectedImage(null);
    setAnswer(null);
    setQuestion("");
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
    >
      <LinearGradient
        colors={[
          theme.colors.gradient2Start,
          theme.colors.gradient2Middle,
          theme.colors.gradient2End,
        ]}
        style={styles.gradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
      >
        {/* Header */}
        <LinearGradient
          colors={[theme.colors.primary, theme.colors.secondary]}
          style={styles.header}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <View style={styles.userInfo}>
            {user?.picture && (
              <Image source={{ uri: user.picture }} style={styles.avatar} />
            )}
            <View>
              <Text style={styles.userName}>{user?.name || "User"}</Text>
              <Text style={styles.userEmail}>{user?.email || ""}</Text>
            </View>
          </View>
          <TouchableOpacity onPress={signOut} style={styles.signOutButton}>
            <MaterialCommunityIcons
              name="logout"
              size={24}
              color={theme.colors.text}
            />
          </TouchableOpacity>
        </LinearGradient>

        <ScrollView
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Title */}
          <Text style={styles.title}>Visual Question Answering</Text>
          <Text style={styles.subtitle}>
            Upload an image and ask a question
          </Text>

          {/* Image Picker */}
          <View style={styles.imageSection}>
            {selectedImage ? (
              <View style={styles.imageContainer}>
                <Image
                  source={{ uri: selectedImage }}
                  style={styles.selectedImage}
                  resizeMode="cover"
                  transition={200}
                  onError={(error) => {
                    console.error("Image load error:", error);
                  }}
                  onLoad={() => {
                    console.log("Image loaded successfully:", selectedImage);
                  }}
                />
                <TouchableOpacity
                  style={styles.clearButton}
                  onPress={clearImage}
                >
                  <MaterialCommunityIcons
                    name="close-circle"
                    size={32}
                    color={theme.colors.error}
                  />
                </TouchableOpacity>
              </View>
            ) : (
              <View style={styles.imagePlaceholder}>
                <MaterialCommunityIcons
                  name="image-plus"
                  size={64}
                  color={theme.colors.textSecondary}
                />
                <Text style={styles.placeholderText}>No image selected</Text>
              </View>
            )}

            <View style={styles.imageButtons}>
              <TouchableOpacity
                style={styles.imageButton}
                onPress={() => pickImage(true)}
              >
                <MaterialCommunityIcons
                  name="camera"
                  size={24}
                  color={theme.colors.text}
                />
                <Text style={styles.imageButtonText}>Camera</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.imageButton}
                onPress={() => pickImage(false)}
              >
                <MaterialCommunityIcons
                  name="image"
                  size={24}
                  color={theme.colors.text}
                />
                <Text style={styles.imageButtonText}>Gallery</Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Question Input */}
          <View style={styles.questionSection}>
            <Text style={styles.sectionTitle}>Your Question</Text>
            <TextInput
              style={styles.questionInput}
              placeholder="What is in the image?"
              placeholderTextColor={theme.colors.textSecondary}
              value={question}
              onChangeText={setQuestion}
              multiline
              maxLength={200}
            />

            <TouchableOpacity
              style={[styles.askButton, loading && styles.askButtonDisabled]}
              onPress={handleAskQuestion}
              disabled={loading}
            >
              {loading ? (
                <ActivityIndicator color={theme.colors.text} />
              ) : (
                <>
                  <MaterialCommunityIcons
                    name="send"
                    size={20}
                    color={theme.colors.text}
                  />
                  <Text style={styles.askButtonText}>Ask Question</Text>
                </>
              )}
            </TouchableOpacity>
          </View>

          {/* Answer Display */}
          {answer && (
            <View style={styles.answerSection}>
              <Text style={styles.sectionTitle}>Answer</Text>
              <View style={styles.answerCard}>
                <View style={styles.answerHeader}>
                  <MaterialCommunityIcons
                    name="lightbulb-on"
                    size={24}
                    color={theme.colors.warning}
                  />
                  <View style={styles.modelBadge}>
                    <Text style={styles.modelBadgeText}>
                      {answer.model_used === "spatial"
                        ? "📍 Spatial Model"
                        : "🔍 Base Model"}
                    </Text>
                  </View>
                </View>

                <Text style={styles.answerText}>{answer.answer}</Text>

                <View style={styles.answerMeta}>
                  <Text style={styles.metaText}>
                    Type: {answer.question_type || "general"}
                  </Text>
                  <Text style={styles.metaText}>
                    Confidence: {(answer.confidence * 100).toFixed(0)}%
                  </Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>
      </LinearGradient>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  gradient: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    padding: theme.spacing.lg,
    paddingTop: theme.spacing.xxl,
    ...theme.shadows.md,
  },
  userInfo: {
    flexDirection: "row",
    alignItems: "center",
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    marginRight: theme.spacing.md,
  },
  userName: {
    color: theme.colors.text,
    fontSize: 16,
    fontWeight: "600",
  },
  userEmail: {
    color: theme.colors.textSecondary,
    fontSize: 12,
  },
  signOutButton: {
    padding: theme.spacing.sm,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: theme.spacing.lg,
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
  },
  subtitle: {
    fontSize: 16,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.xl,
  },
  imageSection: {
    marginBottom: theme.spacing.xl,
  },
  imageContainer: {
    position: "relative",
    borderRadius: theme.borderRadius.lg,
    overflow: "hidden",
    marginBottom: theme.spacing.md,
    backgroundColor: theme.colors.card,
    width: "100%",
    height: 300,
  },
  selectedImage: {
    width: "100%",
    height: 300,
    borderRadius: theme.borderRadius.lg,
    backgroundColor: theme.colors.surface,
  },
  clearButton: {
    position: "absolute",
    top: theme.spacing.md,
    right: theme.spacing.md,
    backgroundColor: "rgba(0, 0, 0, 0.5)",
    borderRadius: theme.borderRadius.full,
  },
  imagePlaceholder: {
    height: 300,
    backgroundColor: theme.colors.card,
    borderRadius: theme.borderRadius.lg,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: theme.spacing.md,
    borderWidth: 2,
    borderColor: theme.colors.surface,
    borderStyle: "dashed",
  },
  placeholderText: {
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.md,
    fontSize: 16,
  },
  imageButtons: {
    flexDirection: "row",
    justifyContent: "space-around",
  },
  imageButton: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: theme.colors.card,
    paddingVertical: theme.spacing.md,
    paddingHorizontal: theme.spacing.xl,
    borderRadius: theme.borderRadius.md,
    ...theme.shadows.sm,
  },
  imageButtonText: {
    color: theme.colors.text,
    marginLeft: theme.spacing.sm,
    fontSize: 16,
    fontWeight: "600",
  },
  questionSection: {
    marginBottom: theme.spacing.xl,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "600",
    color: theme.colors.text,
    marginBottom: theme.spacing.md,
  },
  questionInput: {
    backgroundColor: theme.colors.card,
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
    color: theme.colors.text,
    fontSize: 16,
    minHeight: 100,
    textAlignVertical: "top",
    marginBottom: theme.spacing.md,
  },
  askButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: theme.colors.primary,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
    ...theme.shadows.md,
  },
  askButtonDisabled: {
    opacity: 0.6,
  },
  askButtonText: {
    color: theme.colors.text,
    fontSize: 18,
    fontWeight: "600",
    marginLeft: theme.spacing.sm,
  },
  answerSection: {
    marginBottom: theme.spacing.xl,
  },
  answerCard: {
    backgroundColor: theme.colors.card,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    ...theme.shadows.md,
  },
  answerHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: theme.spacing.md,
  },
  modelBadge: {
    backgroundColor: theme.colors.primary,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.borderRadius.sm,
    marginLeft: theme.spacing.md,
  },
  modelBadgeText: {
    color: theme.colors.text,
    fontSize: 12,
    fontWeight: "600",
  },
  answerText: {
    color: theme.colors.text,
    fontSize: 18,
    lineHeight: 26,
    marginBottom: theme.spacing.md,
  },
  answerMeta: {
    flexDirection: "row",
    justifyContent: "space-between",
    borderTopWidth: 1,
    borderTopColor: theme.colors.surface,
    paddingTop: theme.spacing.md,
  },
  metaText: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
});
