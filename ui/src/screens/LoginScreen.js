import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ActivityIndicator,
  Animated,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { useAuth } from '../contexts/AuthContext';
import { theme } from '../styles/theme';
export default function LoginScreen({ navigation }) {
  const { signIn, signUp } = useAuth();
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const scaleAnim = new Animated.Value(1);
  const handleSubmit = async () => {
    try {
      setLoading(true);
      if (isSignUp) {
        await signUp(email, password, confirmPassword);
      } else {
        await signIn(email, password);
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Authentication failed');
    } finally {
      setLoading(false);
    }
  };
  const animateButton = () => {
    Animated.sequence([
      Animated.timing(scaleAnim, {
        toValue: 0.95,
        duration: 100,
        useNativeDriver: true,
      }),
      Animated.timing(scaleAnim, {
        toValue: 1,
        duration: 100,
        useNativeDriver: true,
      }),
    ]).start();
  };
  return (
    <LinearGradient
      colors={[theme.colors.gradientStart, theme.colors.gradientMiddle, theme.colors.gradientEnd]}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          {}
          <View style={styles.iconContainer}>
            <MaterialCommunityIcons
              name="brain"
              size={80}
              color={theme.colors.text}
            />
          </View>
          {}
          <Text style={styles.title}>VQA Assistant</Text>
          <Text style={styles.subtitle}>Visual Question Answering</Text>
          {}
          <View style={styles.formContainer}>
            <Text style={styles.formTitle}>
              {isSignUp ? 'Create Account' : 'Welcome Back'}
            </Text>
            {}
            <View style={styles.inputContainer}>
              <MaterialCommunityIcons
                name="email"
                size={20}
                color={theme.colors.textSecondary}
                style={styles.inputIcon}
              />
              <TextInput
                style={styles.input}
                placeholder="Email"
                placeholderTextColor={theme.colors.textSecondary}
                value={email}
                onChangeText={setEmail}
                keyboardType="email-address"
                autoCapitalize="none"
                autoCorrect={false}
              />
            </View>
            {}
            <View style={styles.inputContainer}>
              <MaterialCommunityIcons
                name="lock"
                size={20}
                color={theme.colors.textSecondary}
                style={styles.inputIcon}
              />
              <TextInput
                style={styles.input}
                placeholder="Password"
                placeholderTextColor={theme.colors.textSecondary}
                value={password}
                onChangeText={setPassword}
                secureTextEntry={!showPassword}
                autoCapitalize="none"
              />
              <TouchableOpacity
                onPress={() => setShowPassword(!showPassword)}
                style={styles.eyeIcon}
              >
                <MaterialCommunityIcons
                  name={showPassword ? 'eye-off' : 'eye'}
                  size={20}
                  color={theme.colors.textSecondary}
                />
              </TouchableOpacity>
            </View>
            {}
            {isSignUp && (
              <View style={styles.inputContainer}>
                <MaterialCommunityIcons
                  name="lock-check"
                  size={20}
                  color={theme.colors.textSecondary}
                  style={styles.inputIcon}
                />
                <TextInput
                  style={styles.input}
                  placeholder="Confirm Password"
                  placeholderTextColor={theme.colors.textSecondary}
                  value={confirmPassword}
                  onChangeText={setConfirmPassword}
                  secureTextEntry={!showPassword}
                  autoCapitalize="none"
                />
              </View>
            )}
            {}
            <Animated.View style={{ transform: [{ scale: scaleAnim }] }}>
              <TouchableOpacity
                style={styles.submitButton}
                onPress={() => {
                  animateButton();
                  handleSubmit();
                }}
                disabled={loading}
              >
                {loading ? (
                  <ActivityIndicator color={theme.colors.background} />
                ) : (
                  <Text style={styles.submitButtonText}>
                    {isSignUp ? 'Sign Up' : 'Sign In'}
                  </Text>
                )}
              </TouchableOpacity>
            </Animated.View>
            {}
            <TouchableOpacity
              onPress={() => {
                setIsSignUp(!isSignUp);
                setConfirmPassword('');
              }}
              style={styles.toggleButton}
            >
              <Text style={styles.toggleText}>
                {isSignUp
                  ? 'Already have an account? Sign In'
                  : "Don't have an account? Sign Up"}
              </Text>
            </TouchableOpacity>
          </View>
          {}
          <View style={styles.featuresContainer}>
            <View style={styles.feature}>
              <MaterialCommunityIcons name="image-search" size={20} color={theme.colors.text} />
              <Text style={styles.featureText}>Image Analysis</Text>
            </View>
            <View style={styles.feature}>
              <MaterialCommunityIcons name="compass" size={20} color={theme.colors.text} />
              <Text style={styles.featureText}>Spatial Reasoning</Text>
            </View>
            <View style={styles.feature}>
              <MaterialCommunityIcons name="lightning-bolt" size={20} color={theme.colors.text} />
              <Text style={styles.featureText}>Fast Answers</Text>
            </View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </LinearGradient>
  );
}
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    padding: theme.spacing.xl,
  },
  iconContainer: {
    alignSelf: 'center',
    marginBottom: theme.spacing.lg,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: theme.borderRadius.full,
    padding: theme.spacing.lg,
  },
  title: {
    fontSize: 36,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: theme.colors.text,
    marginBottom: theme.spacing.xl,
    textAlign: 'center',
    opacity: 0.9,
  },
  formContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: theme.borderRadius.lg,
    padding: theme.spacing.xl,
    marginBottom: theme.spacing.xl,
  },
  formTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginBottom: theme.spacing.lg,
    textAlign: 'center',
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    borderRadius: theme.borderRadius.md,
    marginBottom: theme.spacing.md,
    paddingHorizontal: theme.spacing.md,
  },
  inputIcon: {
    marginRight: theme.spacing.sm,
  },
  input: {
    flex: 1,
    paddingVertical: theme.spacing.md,
    color: theme.colors.background,
    fontSize: 16,
  },
  eyeIcon: {
    padding: theme.spacing.sm,
  },
  submitButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius.md,
    alignItems: 'center',
    marginTop: theme.spacing.md,
    ...theme.shadows.md,
  },
  submitButtonText: {
    color: theme.colors.background,
    fontSize: 18,
    fontWeight: '600',
  },
  toggleButton: {
    marginTop: theme.spacing.lg,
    alignItems: 'center',
  },
  toggleText: {
    color: theme.colors.text,
    fontSize: 14,
  },
  featuresContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: theme.spacing.lg,
  },
  feature: {
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: theme.borderRadius.md,
    padding: theme.spacing.md,
    minWidth: 90,
  },
  featureText: {
    color: theme.colors.text,
    fontSize: 11,
    marginTop: theme.spacing.sm,
    textAlign: 'center',
  },
});