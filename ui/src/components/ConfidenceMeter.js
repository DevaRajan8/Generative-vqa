import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet } from 'react-native';
import Svg, { Circle, G } from 'react-native-svg';
import { theme } from '../styles/theme';
const AnimatedCircle = Animated.createAnimatedComponent(Circle);
export default function ConfidenceMeter({ confidence, size = 100 }) {
  const animatedValue = useRef(new Animated.Value(0)).current;
  const circleRef = useRef();
  const radius = (size - 10) / 2;
  const circumference = 2 * Math.PI * radius;
  useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: confidence,
      duration: 1000,
      useNativeDriver: true,
    }).start();
  }, [confidence]);
  const strokeDashoffset = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [circumference, 0],
  });
  const getColor = () => {
    if (confidence < 0.5) return theme.colors.error; 
    if (confidence < 0.75) return theme.colors.warning; 
    return theme.colors.success; 
  };
  const percentage = Math.round(confidence * 100);
  return (
    <View style={[styles.container, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        <G rotation="-90" origin={`${size / 2}, ${size / 2}`}>
          {}
          <Circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={theme.colors.surface}
            strokeWidth="8"
            fill="none"
          />
          {}
          <AnimatedCircle
            ref={circleRef}
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={getColor()}
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
          />
        </G>
      </Svg>
      {}
      <View style={styles.textContainer}>
        <Animated.Text style={[styles.percentageText, { color: getColor() }]}>
          {percentage}%
        </Animated.Text>
      </View>
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  textContainer: {
    position: 'absolute',
    justifyContent: 'center',
    alignItems: 'center',
  },
  percentageText: {
    fontSize: 20,
    fontWeight: 'bold',
  },
});