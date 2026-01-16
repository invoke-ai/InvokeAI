import {
  Image,
  Modal,
  ModalBody,
  ModalCloseButton,
  ModalContent,
  ModalHeader,
  ModalOverlay,
  Spinner,
  Text,
} from '@invoke-ai/ui-library';
import { logger } from 'app/logging/logger';
import { useNodeTemplateOrThrow } from 'features/nodes/hooks/useNodeTemplateOrThrow';
import type { ReactElement, ReactNode } from 'react';
import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { marked } from 'marked';

const log = logger('system');

interface NodeDocsContent {
  markdown: string;
  basePath: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

/**
 * Resolves a potentially relative image path to an absolute path based on the docs base path.
 * Handles paths starting with './' or not starting with '/' or 'http'.
 */
const resolveImagePath = (src: string | undefined, basePath: string): string => {
  if (!src) {
    return '';
  }
  // If it's already an absolute URL or data URL, return as-is
  if (src.startsWith('http://') || src.startsWith('https://') || src.startsWith('data:') || src.startsWith('/')) {
    return src;
  }
  // Handle relative paths like './images/...' or 'images/...'
  const relativePath = src.startsWith('./') ? src.slice(2) : src;
  // Normalize path to avoid double slashes
  const normalizedBasePath = basePath.endsWith('/') ? basePath.slice(0, -1) : basePath;
  const normalizedRelativePath = relativePath.startsWith('/') ? relativePath.slice(1) : relativePath;
  return `${normalizedBasePath}/${normalizedRelativePath}`;
};

/**
 * Creates markdown components with proper image path resolution.
 */
// We will not use react-markdown components anymore; keep resolveImagePath for potential future work
const createMarkdownComponents = (basePath: string) => ({
  img: ({ src, alt }: { src?: string; alt?: string }) => (
    <Image src={resolveImagePath(src, basePath)} alt={alt || ''} maxW="100%" my={4} borderRadius="base" />
  ),
});

export const InvocationNodeHelpModal = memo(({ isOpen, onClose }: Props): ReactElement => {
  const nodeTemplate = useNodeTemplateOrThrow();
  const { t, i18n } = useTranslation();
  const [docsContent, setDocsContent] = useState<NodeDocsContent | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      // Reset state when modal closes to prevent stale data
      setDocsContent(null);
      setError(null);
      return;
    }

    const loadDocs = async () => {
      setIsLoading(true);
      setError(null);

      const nodeType = nodeTemplate.type;
      // Sanitize nodeType to prevent path traversal - only allow alphanumeric, underscore, and hyphen
      const sanitizedNodeType = nodeType.replace(/[^a-zA-Z0-9_-]/g, '');
      if (sanitizedNodeType !== nodeType) {
        log.warn({ nodeType }, 'Node type contains invalid characters for docs path');
      }

      const currentLanguage = i18n.language;
      const fallbackLanguage = 'en';
      // Sanitize language code as well
      const sanitizedLanguage = currentLanguage.replace(/[^a-zA-Z-]/g, '');

      // Try to load docs for current language first, then fallback to English
      const languagesToTry =
        sanitizedLanguage !== fallbackLanguage ? [sanitizedLanguage, fallbackLanguage] : [fallbackLanguage];

      for (const lang of languagesToTry) {
        try {
          const basePath = `/nodeDocs/${lang}`;
          const response = await fetch(`${basePath}/${sanitizedNodeType}.md`);
          if (response.ok) {
            const markdown = await response.text();
            setDocsContent({ markdown, basePath });
            setIsLoading(false);
            return;
          }
        } catch {
          // Log error but continue to next language
          log.debug(`Failed to fetch node docs for ${sanitizedNodeType} (${lang})`);
        }
      }

      // No docs found for any language
      setError(t('nodes.noDocsAvailable'));
      setIsLoading(false);
    };

    loadDocs();
  }, [isOpen, nodeTemplate.type, i18n.language, t]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} isCentered size="2xl" useInert={false}>
      <ModalOverlay />
      <ModalContent maxH="80vh">
        <ModalHeader>
          {nodeTemplate.title} - {t('nodes.help')}
        </ModalHeader>
        <ModalCloseButton />
        <ModalBody pb={6} overflowY="auto">
          {isLoading && <Spinner size="lg" />}
          {error && <Text color="base.400">{error}</Text>}
          {docsContent && (
            <div
              // We sanitize by stripping any raw HTML tags from the markdown before rendering
              dangerouslySetInnerHTML={{ __html: marked.parse(docsContent.markdown.replace(/<[^>]+>/g, '')) }}
            />
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

InvocationNodeHelpModal.displayName = 'InvocationNodeHelpModal';
