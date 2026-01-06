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
import { useNodeTemplateOrThrow } from 'features/nodes/hooks/useNodeTemplateOrThrow';
import type { ReactElement, ReactNode } from 'react';
import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import Markdown from 'react-markdown';

interface NodeDocsContent {
  markdown: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

const markdownComponents = {
  // Render images inline with the markdown
  img: ({ src, alt }: { src?: string; alt?: string }) => (
    <Image src={src} alt={alt || ''} maxW="100%" my={4} borderRadius="base" />
  ),
  // Style paragraphs
  p: ({ children }: { children?: ReactNode }) => (
    <Text mb={3} lineHeight="tall">
      {children}
    </Text>
  ),
  // Style headings
  h1: ({ children }: { children?: ReactNode }) => (
    <Text as="h1" fontSize="xl" fontWeight="bold" mt={4} mb={2}>
      {children}
    </Text>
  ),
  h2: ({ children }: { children?: ReactNode }) => (
    <Text as="h2" fontSize="lg" fontWeight="semibold" mt={3} mb={2}>
      {children}
    </Text>
  ),
  h3: ({ children }: { children?: ReactNode }) => (
    <Text as="h3" fontSize="md" fontWeight="semibold" mt={2} mb={1}>
      {children}
    </Text>
  ),
  // Style code blocks
  code: ({ children }: { children?: ReactNode }) => (
    <Text as="code" fontFamily="mono" bg="base.700" px={1} borderRadius="sm">
      {children}
    </Text>
  ),
  // Style list items
  li: ({ children }: { children?: ReactNode }) => (
    <Text as="li" ml={4} mb={1}>
      {children}
    </Text>
  ),
};

export const InvocationNodeHelpModal = memo(({ isOpen, onClose }: Props): ReactElement => {
  const nodeTemplate = useNodeTemplateOrThrow();
  const { t, i18n } = useTranslation();
  const [docsContent, setDocsContent] = useState<NodeDocsContent | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const loadDocs = async () => {
      setIsLoading(true);
      setError(null);

      const nodeType = nodeTemplate.type;
      const currentLanguage = i18n.language;
      const fallbackLanguage = 'en';

      // Try to load docs for current language first, then fallback to English
      const languagesToTry =
        currentLanguage !== fallbackLanguage ? [currentLanguage, fallbackLanguage] : [fallbackLanguage];

      for (const lang of languagesToTry) {
        try {
          const response = await fetch(`/nodeDocs/${lang}/${nodeType}.md`);
          if (response.ok) {
            const markdown = await response.text();
            setDocsContent({ markdown });
            setIsLoading(false);
            return;
          }
        } catch {
          // Continue to next language
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
          {docsContent && <Markdown components={markdownComponents}>{docsContent.markdown}</Markdown>}
        </ModalBody>
      </ModalContent>
    </Modal>
  );
});

InvocationNodeHelpModal.displayName = 'InvocationNodeHelpModal';
