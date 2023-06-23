import { motion } from 'framer-motion';

export const SelectedItemOverlay = () => (
  <motion.div
    initial={{
      opacity: 0,
    }}
    animate={{
      opacity: 1,
      transition: { duration: 0.1 },
    }}
    exit={{
      opacity: 0,
      transition: { duration: 0.1 },
    }}
    style={{
      position: 'absolute',
      top: 0,
      insetInlineStart: 0,
      width: '100%',
      height: '100%',
      boxShadow: 'inset 0px 0px 0px 2px var(--invokeai-colors-accent-300)',
      borderRadius: 'var(--invokeai-radii-base)',
    }}
  />
);
