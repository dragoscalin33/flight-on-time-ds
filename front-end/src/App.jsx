import Header from "./components/Header";
import BuscaVoos from "./pages/BuscaVoos";
import Footer from "./components/Footer";
import ChatWidget from "./components/chat/ChatWidget";

export default function App() {
  return (
    <>
      <Header />
      <BuscaVoos />
      <Footer />
      <ChatWidget />
    </>
  );
}
