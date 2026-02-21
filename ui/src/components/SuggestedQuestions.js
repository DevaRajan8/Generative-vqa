import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { theme } from '../styles/theme';
const SUGGESTED_QUESTIONS = [
  {
    category: 'Objects',
    icon: 'cube-outline',
    questions: [
      'What objects are in the image?',
      'What is the main object?',
      'How many objects are there?',
    ],
  },
  {
    category: 'Colors',
    icon: 'palette',
    questions: [
      'What color is the object?',
      'What colors are in the image?',
      'What is the dominant color?',
    ],
  },
  {
    category: 'Counting',
    icon: 'counter',
    questions: [
      'How many people are there?',
      'How many items are visible?',
      'What is the count of objects?',
    ],
  },
  {
    category: 'Spatial',
    icon: 'compass',
    questions: [
      'What is on the left?',
      'What is on the right?',
      'What is in the center?',
      'What is above the object?',
      'What is below the object?',
    ],
  },
  {
    category: 'Actions',
    icon: 'run',
    questions: [
      'What is happening in the image?',
      'What is the person doing?',
      'What activity is shown?',
    ],
  },
  {
    category: 'Scene',
    icon: 'image-multiple',
    questions: [
      'Where is this photo taken?',
      'What type of scene is this?',
      'Is this indoors or outdoors?',
    ],
  },
];
export default function SuggestedQuestions({ onQuestionSelect }) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <MaterialCommunityIcons 
          name="lightbulb-on" 
          size={20} 
          color={theme.colors.primary} 
        />
        <Text style={styles.headerText}>Suggested Questions</Text>
      </View>
      <ScrollView 
        horizontal 
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {SUGGESTED_QUESTIONS.map((category, categoryIndex) => (
          <View key={categoryIndex} style={styles.categoryContainer}>
            <View style={styles.categoryHeader}>
              <MaterialCommunityIcons 
                name={category.icon} 
                size={16} 
                color={theme.colors.textSecondary} 
              />
              <Text style={styles.categoryText}>{category.category}</Text>
            </View>
            {category.questions.map((question, questionIndex) => (
              <TouchableOpacity
                key={questionIndex}
                style={styles.chip}
                onPress={() => onQuestionSelect(question)}
                activeOpacity={0.7}
              >
                <Text style={styles.chipText} numberOfLines={1}>
                  {question}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        ))}
      </ScrollView>
    </View>
  );
}
const styles = StyleSheet.create({
  container: {
    marginBottom: theme.spacing.lg,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.md,
  },
  headerText: {
    fontSize: 16,
    fontWeight: '600',
    color: theme.colors.text,
    marginLeft: theme.spacing.sm,
  },
  scrollContent: {
    paddingRight: theme.spacing.lg,
  },
  categoryContainer: {
    marginRight: theme.spacing.lg,
  },
  categoryHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: theme.spacing.sm,
  },
  categoryText: {
    fontSize: 12,
    fontWeight: '600',
    color: theme.colors.textSecondary,
    marginLeft: theme.spacing.xs,
    textTransform: 'uppercase',
  },
  chip: {
    backgroundColor: theme.colors.card,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderRadius: theme.borderRadius.full,
    marginBottom: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.primary,
    ...theme.shadows.sm,
  },
  chipText: {
    color: theme.colors.text,
    fontSize: 14,
    maxWidth: 200,
  },
});