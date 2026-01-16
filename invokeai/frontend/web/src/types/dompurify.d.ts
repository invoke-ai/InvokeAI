declare module 'dompurify' {
  interface DOMPurifyInstance {
    sanitize: (dirty: string) => string;
  }
  const DOMPurify: DOMPurifyInstance;
  export default DOMPurify;
}
