import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { SearchPage } from "./pages/search";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SearchPage />
  </StrictMode>,
);
