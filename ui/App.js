import React, { useEffect, useState } from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import { StatusBar } from "expo-status-bar";
import { Platform, LogBox } from "react-native";
import { AuthProvider, useAuth } from "./src/contexts/AuthContext";
import LoginScreen from "./src/screens/LoginScreen";
import HomeScreen from "./src/screens/HomeScreen";
import { theme } from "./src/styles/theme";
LogBox.ignoreLogs([
  "Property 'ipconfig' doesn't exist", 
  "ImagePicker.MediaTypeOptions", 
]);
const Stack = createNativeStackNavigator();
if (Platform.OS !== "web") {
  import("expo-splash-screen").then(({ default: SplashScreen }) => {
    SplashScreen.preventAutoHideAsync();
  });
}
function AppNavigator() {
  const { user, loading } = useAuth();
  const [appReady, setAppReady] = useState(false);
  useEffect(() => {
    if (Platform.OS === "web" || !loading) {
      setAppReady(true);
    }
  }, [loading]);
  if (!appReady) {
    return (
      <></> 
    );
  }
  return (
    <>
      <StatusBar style="light" />
      <NavigationContainer>
        <Stack.Navigator
          screenOptions={{
            headerShown: false,
            contentStyle: { backgroundColor: theme.colors.background },
          }}
        >
          {user ? (
            <Stack.Screen name="Home" component={HomeScreen} />
          ) : (
            <Stack.Screen name="Login" component={LoginScreen} />
          )}
        </Stack.Navigator>
      </NavigationContainer>
    </>
  );
}
export default function App() {
  return (
    <AuthProvider>
      <AppNavigator />
    </AuthProvider>
  );
}