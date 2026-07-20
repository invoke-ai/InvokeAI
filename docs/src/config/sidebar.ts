import type { StarlightUserConfig } from '@astrojs/starlight/types';
import { makeChangelogsSidebarLinks } from 'starlight-changelogs';

type SidebarConfig = StarlightUserConfig['sidebar'];

const sidebar: SidebarConfig = [
  {
    label: 'Start Here',
    translations: {
      de: 'Erste Schritte',
      es: 'Primeros pasos',
      hi: 'शुरू करें',
    },
    items: [
      {
        autogenerate: { directory: 'start-here' },
      },
    ],
  },
  {
    label: 'Configuration',
    translations: {
      de: 'Konfiguration',
      es: 'Configuración',
      hi: 'कॉन्फ़िगरेशन',
    },
    items: [
      {
        autogenerate: { directory: 'configuration' },
      },
    ],
  },
  {
    label: 'Concepts',
    translations: {
      de: 'Konzepte',
      es: 'Conceptos',
      hi: 'अवधारणाएँ',
    },
    items: [
      {
        autogenerate: { directory: 'concepts' },
      },
    ],
  },
  {
    label: 'Features',
    translations: {
      de: 'Funktionen',
      es: 'Funciones',
      hi: 'सुविधाएँ',
    },
    items: [
      {
        autogenerate: { directory: 'features' },
      },
    ],
  },
  {
    label: 'Development',
    translations: {
      de: 'Entwicklung',
      es: 'Desarrollo',
      hi: 'विकास',
    },
    items: [
      {
        autogenerate: { directory: 'development', collapsed: true },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Contributing',
    translations: {
      de: 'Mitwirken',
      es: 'Contribuir',
      hi: 'योगदान',
    },
    items: [
      {
        autogenerate: { directory: 'contributing' },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Troubleshooting & Help',
    translations: {
      de: 'Fehlerbehebung & Hilfe',
      es: 'Solución de problemas y ayuda',
      hi: 'समस्या निवारण और सहायता',
    },
    items: [
      {
        autogenerate: { directory: 'troubleshooting' },
      },
    ],
    collapsed: true,
  },
  {
    label: 'Releases',
    translations: {
      de: 'Versionen',
      es: 'Versiones',
      hi: 'रिलीज़',
    },
    collapsed: true,
    items: [
      ...makeChangelogsSidebarLinks([
        {
          type: 'recent',
          base: 'releases',
        },
      ]),
    ],
  },
];

export { sidebar as sidebarConfig };
