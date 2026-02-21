import React, { useState, useEffect } from 'react';
import { Text, Animated } from 'react-native';
export default function TypingText({ 
  text, 
  style, 
  speed = 30, 
  onComplete,
  showCursor = true 
}) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);
  const cursorOpacity = new Animated.Value(1);
  useEffect(() => {
    setDisplayedText('');
    setCurrentIndex(0);
    setIsComplete(false);
  }, [text]);
  useEffect(() => {
    if (currentIndex < text.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(prev => prev + text[currentIndex]);
        setCurrentIndex(prev => prev + 1);
      }, speed);
      return () => clearTimeout(timeout);
    } else if (currentIndex === text.length && !isComplete) {
      setIsComplete(true);
      if (onComplete) onComplete();
      if (showCursor) {
        Animated.loop(
          Animated.sequence([
            Animated.timing(cursorOpacity, {
              toValue: 0,
              duration: 500,
              useNativeDriver: true,
            }),
            Animated.timing(cursorOpacity, {
              toValue: 1,
              duration: 500,
              useNativeDriver: true,
            }),
          ])
        ).start();
      }
    }
  }, [currentIndex, text, speed, isComplete]);
  return (
    <Text style={style}>
      {displayedText}
      {showCursor && isComplete && (
        <Animated.Text style={[style, { opacity: cursorOpacity }]}>
          |
        </Animated.Text>
      )}
    </Text>
  );
}