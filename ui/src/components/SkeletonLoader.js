import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { theme } from '../styles/theme';
export default function SkeletonLoader({ 
  variant = 'text', 
  width = '100%', 
  height = 20,
  style 
}) {
  const animatedValue = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(animatedValue, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: true,
        }),
        Animated.timing(animatedValue, {
          toValue: 0,
          duration: 1500,
          useNativeDriver: true,
        }),
      ])
    ).start();
  }, []);
  const opacity = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 0.7],
  });
  const getSkeletonStyle = () => {
    switch (variant) {
      case 'circular':
        return {
          width: width,
          height: width,
          borderRadius: width / 2,
        };
      case 'card':
        return {
          width: width,
          height: height,
          borderRadius: theme.borderRadius.lg,
        };
      case 'image':
        return {
          width: width,
          height: height,
          borderRadius: theme.borderRadius.md,
        };
      case 'text':
      default:
        return {
          width: width,
          height: height,
          borderRadius: theme.borderRadius.sm,
        };
    }
  };
  return (
    <Animated.View 
      style={[
        styles.skeleton, 
        getSkeletonStyle(), 
        { opacity },
        style
      ]}
    >
      <LinearGradient
        colors={[
          theme.colors.surface,
          theme.colors.card,
          theme.colors.surface,
        ]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 0 }}
        style={StyleSheet.absoluteFill}
      />
    </Animated.View>
  );
}
const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: theme.colors.surface,
    overflow: 'hidden',
  },
});