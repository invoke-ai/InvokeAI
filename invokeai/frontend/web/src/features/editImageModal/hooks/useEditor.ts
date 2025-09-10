import { Editor } from "features/editImageModal/lib/editor";
import type { RefObject} from "react";
import { useEffect,useState } from "react";

export const useEditor = (arg: { containerRef: RefObject<HTMLDivElement | null> }) => {
    const editor = useState(() => new Editor())[0];

    useEffect(() => {
        const container = arg.containerRef.current;
        if (container) {
            editor.init(container);
            
            // Handle window resize
            const handleResize = () => {
                editor.resize(container.clientWidth, container.clientHeight);
            };
            
            window.addEventListener("resize", handleResize);
            return () => {
                window.removeEventListener("resize", handleResize);
            };
        }
    }, [arg.containerRef, editor]);

    // Clean up editor on unmount
    useEffect(() => {
        return () => {
            editor.destroy();
        };
    }, [editor]);

    return editor;
};
