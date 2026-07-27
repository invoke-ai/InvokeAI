// The dialog body is deliberately absent: it is reached only through
// `SettingsButton`'s lazy import, and re-exporting it here would make every
// barrel consumer load it eagerly.
export * from './SettingsButton';
export * from './settingsDialogStore';
export * from './store';
