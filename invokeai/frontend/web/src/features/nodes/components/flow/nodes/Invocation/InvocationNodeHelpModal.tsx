import {
  Box,
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
import DOMPurify from 'dompurify';
import { useNodeTemplateOrThrow } from 'features/nodes/hooks/useNodeTemplateOrThrow';
import { marked } from 'marked';
import { memo, type ReactElement, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';

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
 * Rewrite relative image paths in markdown to be absolute based on basePath
 */
const rewriteRelativeImagePaths = (markdown: string, basePath: string): string => {
  return markdown.replace(/!\[([^\]]*)\]\((?!\s*(?:https?:\/\/|\/|data:))([^)]+)\)/g, (_match, alt, src) => {
    const cleaned = src.startsWith('./') ? src.slice(2) : src;
    const normalized = cleaned.startsWith('/') ? cleaned.slice(1) : cleaned;
    return `![${alt}](${basePath}/${normalized})`;
  });
};

/**
 * Creates markdown components with proper image path resolution.
 */
// We will not use react-markdown components anymore; keep resolveImagePath for potential future work
const _createMarkdownComponents = (basePath: string) => ({
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
  const [sanitizedHtml, setSanitizedHtml] = useState<string | null>(null);

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

  useEffect(() => {
    if (!docsContent) {
      setSanitizedHtml(null);
      return;
    }

    let mounted = true;
    (async () => {
      const htmlOrPromise = marked.parse(rewriteRelativeImagePaths(docsContent.markdown, docsContent.basePath));
      const html = typeof htmlOrPromise === 'string' ? htmlOrPromise : await htmlOrPromise;
      if (!mounted) {
        return;
      }
      setSanitizedHtml(DOMPurify.sanitize(html));
    })();

    return () => {
      mounted = false;
    };
  }, [docsContent]);

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
          {sanitizedHtml && (
            <Box
              className="markdown-body"
              style={{ maxWidth: '100%' }}
              bg="transparent"
              color="base.100"
              sx={{
                // Headings
                h1: { color: 'base.100', fontSize: '2xl', mt: 2, mb: 2, fontWeight: 'semibold' },
                h2: { color: 'base.100', fontSize: 'xl', mt: 2, mb: 2, fontWeight: 'semibold' },
                h3: { color: 'base.100', fontSize: 'lg', mt: 2, mb: 1.5, fontWeight: 'semibold' },
                h4: { color: 'base.100', fontSize: 'md', mt: 1.5, mb: 1, fontWeight: 'semibold' },

                // Paragraphs
                p: { color: 'base.100', mt: 1, mb: 1 },

                // Links
                a: { color: 'blue.200', _hover: { textDecoration: 'underline', color: 'blue.300' } },

                // Lists
                ul: { pl: 6, mt: 1, mb: 1 },
                ol: { pl: 6, mt: 1, mb: 1 },
                li: { mt: 1, mb: 1 },

                // Code
                pre: {
                  bg: 'base.800',
                  color: 'base.100',
                  borderRadius: '6px',
                  px: 4,
                  py: 3,
                  overflowX: 'auto',
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, 'Roboto Mono', 'Courier New', monospace",
                  fontSize: 'sm',
                },
                code: {
                  bg: 'rgba(255,255,255,0.02)',
                  color: 'base.100',
                  px: '0.25rem',
                  py: '0.125rem',
                  borderRadius: '4px',
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, 'Roboto Mono', 'Courier New', monospace",
                  fontSize: '0.85em',
                },

                // Blockquote
                blockquote: {
                  borderLeft: '4px solid rgba(255,255,255,0.06)',
                  background: 'rgba(255,255,255,0.02)',
                  color: 'base.200',
                  py: 2,
                  px: 4,
                  my: 3,
                },

                // Tables
                table: { width: '100%', borderCollapse: 'collapse', my: 2 },
                th: {
                  border: '1px solid rgba(255,255,255,0.06)',
                  px: 3,
                  py: 2,
                  textAlign: 'left',
                  color: 'base.100',
                },
                td: {
                  border: '1px solid rgba(255,255,255,0.06)',
                  px: 3,
                  py: 2,
                  color: 'base.100',
                },

                // Images
                img: { maxW: '100%', borderRadius: '6px', display: 'block', my: 3 },

                // Horizontal rule
                hr: { border: 'none', h: '1px', bg: 'base.700', my: 4 },
              }}
              // Render sanitized HTML
              dangerouslySetInnerHTML={{ __html: sanitizedHtml }}
            />
          )}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

InvocationNodeHelpModal.displayName = 'InvocationNodeHelpModal';
