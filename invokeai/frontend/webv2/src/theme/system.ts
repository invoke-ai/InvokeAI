import { createSystem, defaultConfig, defineConfig } from '@chakra-ui/react';

/**
 * Workbench design tokens.
 *
 * The v7 workbench is a dark-only creative surface (Photoshop / Blender / DaVinci
 * feel), so the palette is expressed as concrete tokens and surfaced through
 * semantic tokens with single values. Components reference the semantic tokens
 * (`bg.shell`, `fg.muted`, `accent.invoke`, …) rather than raw hex, which keeps
 * the shell theme-able and removes the brittle hard-coded colors the prototype
 * relied on.
 */
const config = defineConfig({
  globalCss: {
    'html, body, #root': {
      height: '100%',
    },
    body: {
      margin: 0,
      minWidth: '960px',
      minHeight: '720px',
      overflow: 'hidden',
      bg: 'bg.shell',
      color: 'fg.default',
      fontFamily: 'body',
    },
  },
  theme: {
    tokens: {
      fonts: {
        body: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        },
        heading: {
          value: "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif",
        },
      },
      colors: {
        workbench: {
          shell: { value: '#141514' },
          surface: { value: '#161716' },
          surfaceRaised: { value: '#181918' },
          center: { value: '#1a1b1a' },
          canvas: { value: '#1c1d1c' },
          panel: { value: '#282928' },
          panelStroke: { value: '#303330' },
          line: { value: '#2e312d' },
          lineStrong: { value: '#343832' },
          dot: { value: '#343734' },
          fg: { value: '#d7dec7' },
          fgMuted: { value: '#a9afa1' },
          fgSubtle: { value: '#6f756a' },
        },
        lime: {
          accent: { value: '#cbff63' },
          contrast: { value: '#10160b' },
          mutedBg: { value: '#33372a' },
          mutedFg: { value: '#dfff8c' },
        },
        sky: {
          accent: { value: '#59cfff' },
          contrast: { value: '#081218' },
        },
      },
    },
    semanticTokens: {
      colors: {
        'bg.shell': { value: '{colors.workbench.shell}' },
        'bg.surface': { value: '{colors.workbench.surface}' },
        'bg.surfaceRaised': { value: '{colors.workbench.surfaceRaised}' },
        'bg.center': { value: '{colors.workbench.center}' },
        'bg.canvas': { value: '{colors.workbench.canvas}' },
        'bg.panel': { value: '{colors.workbench.panel}' },
        'border.subtle': { value: '{colors.workbench.line}' },
        'border.emphasis': { value: '{colors.workbench.lineStrong}' },
        'border.panel': { value: '{colors.workbench.panelStroke}' },
        'fg.default': { value: '{colors.workbench.fg}' },
        'fg.muted': { value: '{colors.workbench.fgMuted}' },
        'fg.subtle': { value: '{colors.workbench.fgSubtle}' },
        'accent.invoke': { value: '{colors.lime.accent}' },
        'accent.invokeFg': { value: '{colors.lime.contrast}' },
        'accent.widget': { value: '{colors.lime.mutedBg}' },
        'accent.widgetFg': { value: '{colors.lime.mutedFg}' },
        'accent.active': { value: '{colors.sky.accent}' },
        'accent.activeFg': { value: '{colors.sky.contrast}' },
        'canvas.dot': { value: '{colors.workbench.dot}' },
      },
    },
  },
});

export const system = createSystem(defaultConfig, config);
