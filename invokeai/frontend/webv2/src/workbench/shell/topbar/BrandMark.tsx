import { Link } from '@tanstack/react-router';
import { DEFAULT_THEME_ID, THEMES_BY_ID } from '@theme/themes';
import { useWorkbenchPreferences } from '@workbench/settings/store';

/** Compact Invoke logo used as a link to home screen. */
export const BrandMark = () => {
  const { themeId } = useWorkbenchPreferences();
  const theme = THEMES_BY_ID[themeId] ?? THEMES_BY_ID[DEFAULT_THEME_ID];

  return (
    <Link
      to="/"
      style={{
        alignItems: 'center',
        aspectRatio: '1/1',
        display: 'flex',
        flexShrink: 0,
        height: '100%',
        justifyContent: 'center',
      }}
    >
      <svg aria-hidden="true" fill="none" height="20" viewBox="0 0 44 44" width="20">
        <path
          d="M29.1951 10.6667H42V2H2V10.6667H14.8049L29.1951 33.3333H42V42H2V33.3333H14.8049"
          stroke={theme.colors.brand.solid}
          strokeWidth="2.8"
        />
      </svg>
    </Link>
  );
};
