import React, { createContext, useState, useContext, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
const AuthContext = createContext({});
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    loadUser();
  }, []);
  const loadUser = async () => {
    try {
      const userData = await AsyncStorage.getItem('user');
      if (userData) {
        setUser(JSON.parse(userData));
      }
    } catch (error) {
      console.error('Error loading user:', error);
    } finally {
      setLoading(false);
    }
  };
  const signIn = async (email, password) => {
    try {
      if (!email || !password) {
        throw new Error('Email and password are required');
      }
      if (!email.includes('@')) {
        throw new Error('Please enter a valid email address');
      }
      if (password.length < 4) {
        throw new Error('Password must be at least 4 characters');
      }
      const userData = {
        email: email,
        name: email.split('@')[0], 
        picture: null, 
      };
      setUser(userData);
      await AsyncStorage.setItem('user', JSON.stringify(userData));
      return { success: true };
    } catch (error) {
      console.error('Error signing in:', error);
      throw error;
    }
  };
  const signUp = async (email, password, confirmPassword) => {
    try {
      if (!email || !password || !confirmPassword) {
        throw new Error('All fields are required');
      }
      if (!email.includes('@')) {
        throw new Error('Please enter a valid email address');
      }
      if (password.length < 4) {
        throw new Error('Password must be at least 4 characters');
      }
      if (password !== confirmPassword) {
        throw new Error('Passwords do not match');
      }
      return await signIn(email, password);
    } catch (error) {
      console.error('Error signing up:', error);
      throw error;
    }
  };
  const signOut = async () => {
    try {
      await AsyncStorage.removeItem('user');
      setUser(null);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };
  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        signIn,
        signUp,
        signOut,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};