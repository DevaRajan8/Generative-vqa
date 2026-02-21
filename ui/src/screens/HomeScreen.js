import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from "react-native";
import { Image } from "expo-image";
import { LinearGradient } from "expo-linear-gradient";
import { BlurView } from "expo-blur";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import { useAuth } from "../contexts/AuthContext";
import { askQuestion } from "../services/api";
import { theme } from "../styles/theme";
import ConfidenceMeter from "../components/ConfidenceMeter";
import TypingText from "../components/TypingText";
import SkeletonLoader from "../components/SkeletonLoader";
import SuggestedQuestions from "../components/SuggestedQuestions";
export default function HomeScreen() {
  const { user, signOut } = useAuth();
  const [selectedImage, setSelectedImage] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [loading, setLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const pickImage = async (useCamera = false) => {
    try {
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
      const result = useCamera
        ? await ImagePicker.launchCameraAsync({
            mediaTypes: ["images"], 
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
          })
        : await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ["images"], 
            allowsEditing: true,
            aspect: [4, 3],
            quality: 0.8,
          });
      if (!result.canceled && result.assets && result.assets.length > 0) {
        const asset = result.assets[0];
        console.log("Image selected:", asset.uri);
        console.log("Image details:", asset);
        setSelectedImage(asset);
        setAnswer(null); 
      }
    } catch (error) {
      console.error("Error picking image:", error);
      Alert.alert("Error", "Failed to pick image: " + error.message);
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
      const result = await askQuestion(selectedImage.uri, question);
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
    if (isSpeaking) {
      Speech.stop();
      setIsSpeaking(false);
    }
  };
  const handleSpeak = (text) => {
    if (isSpeaking) {
      Speech.stop();
      setIsSpeaking(false);
    } else {
      setIsSpeaking(true);
      Speech.speak(text, {
        language: "en",
        pitch: 1.0,
        rate: 0.9,
        onDone: () => setIsSpeaking(false),
        onStopped: () => setIsSpeaking(false),
        onError: () => setIsSpeaking(false),
      });
    }
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
        {}
        <LinearGradient
          colors={[theme.colors.primary, theme.colors.secondary]}
          style={styles.header}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <View style={styles.userInfo}>
            {user?.picture && (
              <Image
                source={{ uri: user.picture }}
                style={styles.avatar}
                cachePolicy="memory-disk"
              />
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
          {}
          <Text style={styles.title}>Visual Question Answering</Text>
          <Text style={styles.subtitle}>
            Upload an image and ask a question
          </Text>
          {}
          <View style={styles.imageSection}>
            {selectedImage ? (
              <View style={styles.imageContainer}>
                <Image
                  source={{ uri: selectedImage.uri }}
                  style={styles.selectedImage}
                  contentFit="contain"
                  cachePolicy="memory-disk"
                  priority="high"
                  onError={(error) => {
                    console.error("Image load error:", error);
                    Alert.alert("Error", "Failed to load image");
                  }}
                  onLoad={() => {
                    console.log(
                      "Image loaded successfully:",
                      selectedImage.uri,
                    );
                  }}
                />
                <TouchableOpacity
                  style={styles.clearButton}
                  onPress={clearImage}
                >
                  <MaterialCommunityIcons
                    name="close-circle"
                    size={32}
                    color="#FF6B6B"
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
          {}
          <SuggestedQuestions onQuestionSelect={setQuestion} />
          {}
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
                <View style={styles.loadingContainer}>
                  <SkeletonLoader variant="text" width="60%" height={20} style={{ marginBottom: 8 }} />
                  <SkeletonLoader variant="text" width="80%" height={20} />
                </View>
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
          {}
          {answer && (
            <View style={styles.answerSection}>
              <BlurView intensity={80} tint="dark" style={styles.answerCard}>
                <View style={styles.answerHeader}>
                  <MaterialCommunityIcons
                    name="lightbulb"
                    size={24}
                    color={theme.colors.primary}
                  />
                  <Text style={styles.sectionTitle}>Answer</Text>
                  {answer.model && (
                    <View style={styles.modelBadge}>
                      <Text style={styles.modelBadgeText}>{answer.model}</Text>
                    </View>
                  )}
                </View>
                <TypingText 
                  text={answer.answer} 
                  style={styles.answerText}
                  speed={theme.animations.typingSpeed}
                  showCursor={false}
                />
                <View style={styles.answerMeta}>
                  {answer.processing_time && (
                    <Text style={styles.metaText}>
                      ⚡ {answer.processing_time.toFixed(2)}s
                    </Text>
                  )}
                  {answer.confidence && (
                    <View style={styles.confidenceContainer}>
                      <ConfidenceMeter 
                        confidence={answer.confidence} 
                        size={60}
                      />
                    </View>
                  )}
                </View>
              </BlurView>
              {answer.description && (
                <BlurView intensity={80} tint="dark" style={styles.descriptionCard}>
                  <View style={styles.descriptionHeader}>
                    <View style={styles.descriptionTitleRow}>
                      <MaterialCommunityIcons
                        name="text-to-speech"
                        size={24}
                        color={theme.colors.success}
                      />
                      <Text style={styles.descriptionTitle}>
                        Accessible Description
                      </Text>
                    </View>
                    <TouchableOpacity
                      style={styles.speakButton}
                      onPress={() => handleSpeak(answer.description)}
                    >
                      <MaterialCommunityIcons
                        name={isSpeaking ? "stop-circle" : "volume-high"}
                        size={28}
                        color={
                          isSpeaking ? theme.colors.error : theme.colors.primary
                        }
                      />
                    </TouchableOpacity>
                  </View>
                  <TypingText 
                    text={answer.description} 
                    style={styles.descriptionText}
                    speed={theme.animations.typingSpeed}
                    showCursor={false}
                  />
                  {answer.description_status &&
                    answer.description_status !== "success" && (
                      <Text style={styles.descriptionStatus}>
                        ℹ️ Using {answer.description_status} mode
                      </Text>
                    )}
                </BlurView>
              )}
              {}
              {answer.kg_enhancement && (
                <BlurView intensity={80} tint="dark" style={styles.kgCard}>
                  <View style={styles.kgHeader}>
                    <MaterialCommunityIcons
                      name="brain"
                      size={24}
                      color={theme.colors.info}
                    />
                    <Text style={styles.kgTitle}>Common-Sense Reasoning</Text>
                    {answer.reasoning_type === 'neuro-symbolic' && (
                      <View style={styles.neurosymbolicBadge}>
                        <Text style={styles.badgeText}>🧠+🔗</Text>
                      </View>
                    )}
                  </View>
                  <TypingText 
                    text={answer.kg_enhancement} 
                    style={styles.kgText}
                    speed={theme.animations.typingSpeed}
                    showCursor={false}
                  />
                  <View style={styles.kgFooter}>
                    <Text style={styles.kgFooterText}>
                      💡 Enhanced with Knowledge Graph
                    </Text>
                  </View>
                </BlurView>
              )}
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
    backgroundColor: "#FFFFFF", 
    width: "100%",
    minHeight: 300,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: theme.colors.primary,
  },
  selectedImage: {
    width: "100%",
    height: 300,
    minHeight: 300,
  },
  clearButton: {
    position: "absolute",
    top: theme.spacing.md,
    right: theme.spacing.md,
    backgroundColor: "rgba(0, 0, 0, 0.6)",
    borderRadius: 20,
    padding: 4,
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
  loadingContainer: {
    width: '100%',
    alignItems: 'center',
  },
  answerSection: {
    marginBottom: theme.spacing.xl,
  },
  answerCard: {
    backgroundColor: theme.colors.glassBackground,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    overflow: 'hidden',
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
    alignItems: "center",
    borderTopWidth: 1,
    borderTopColor: theme.colors.surface,
    paddingTop: theme.spacing.md,
  },
  metaText: {
    color: theme.colors.textSecondary,
    fontSize: 14,
  },
  confidenceContainer: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  descriptionCard: {
    backgroundColor: theme.colors.glassBackground,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    marginTop: theme.spacing.md,
    borderLeftWidth: 4,
    borderLeftColor: theme.colors.success,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    overflow: 'hidden',
    ...theme.shadows.md,
  },
  descriptionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: theme.spacing.md,
  },
  descriptionTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    flex: 1,
  },
  descriptionTitle: {
    fontSize: 18,
    fontWeight: "600",
    color: theme.colors.text,
    marginLeft: theme.spacing.sm,
  },
  descriptionText: {
    color: theme.colors.text,
    fontSize: 16,
    lineHeight: 24,
    marginBottom: theme.spacing.sm,
  },
  descriptionStatus: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    fontStyle: "italic",
    marginTop: theme.spacing.sm,
  },
  speakButton: {
    padding: theme.spacing.sm,
    borderRadius: theme.borderRadius.full,
    backgroundColor: theme.colors.surface,
  },
  kgCard: {
    backgroundColor: theme.colors.glassBackground,
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.lg,
    marginTop: theme.spacing.md,
    borderLeftWidth: 4,
    borderLeftColor: theme.colors.info,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    overflow: 'hidden',
    ...theme.shadows.md,
  },
  kgHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  kgTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: theme.colors.text,
    marginLeft: theme.spacing.sm,
    flex: 1,
  },
  neurosymbolicBadge: {
    backgroundColor: theme.colors.info,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.borderRadius.full,
  },
  badgeText: {
    fontSize: 14,
    fontWeight: '600',
  },
  kgText: {
    color: theme.colors.text,
    fontSize: 16,
    lineHeight: 24,
    marginBottom: theme.spacing.md,
  },
  kgFooter: {
    borderTopWidth: 1,
    borderTopColor: theme.colors.surface,
    paddingTop: theme.spacing.sm,
  },
  kgFooterText: {
    color: theme.colors.textSecondary,
    fontSize: 12,
    fontStyle: 'italic',
  },
});