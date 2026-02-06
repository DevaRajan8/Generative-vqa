# Icon Placeholder

The app needs custom icons. You can:

1. **Use default Expo icons** (current setup)
2. **Create custom icons** using:
   - [Figma](https://www.figma.com/)
   - [Canva](https://www.canva.com/)
   - AI image generators (when available)

## Required Icons

### icon.png
- Size: 1024x1024 pixels
- Format: PNG
- Design: Brain + eye/image symbol with gradient (indigo to pink)
- Used for: App icon on home screen

### splash.png
- Size: 1242x2436 pixels (iPhone X resolution)
- Format: PNG
- Design: Gradient background with centered logo and "VQA Assistant" text
- Used for: App splash screen on launch

### adaptive-icon.png (Android)
- Size: 1024x1024 pixels
- Format: PNG
- Design: Same as icon.png but with safe zone
- Used for: Android adaptive icon

### favicon.png (Web)
- Size: 48x48 pixels
- Format: PNG
- Design: Simplified version of main icon
- Used for: Web version favicon

## Temporary Solution

For now, the app will use Expo's default icons. To add custom icons:

1. Create the icons with the specifications above
2. Save them in the `ui/assets/` folder
3. Update `app.json` if needed
4. Restart the Expo development server

## Design Guidelines

- **Colors**: Use gradient from #6366F1 (indigo) to #EC4899 (pink)
- **Style**: Modern, minimalist, professional
- **Symbol**: Brain (AI) + Eye/Image (visual) + Question mark (optional)
- **Background**: Transparent or gradient
