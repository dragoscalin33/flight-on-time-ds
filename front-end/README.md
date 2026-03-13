# FlightOnTime Frontend

**Flight On Time — Dashboard** is the frontend interface that allows users to check flight delay predictions based on origin, destination, airline, and date/time.

Built with **React + Vite + Tailwind CSS**, it communicates with the backend API to predict delays. The interface is responsive, interactive, and features **autocomplete for airports and airlines**, along with IATA code validation.

---

## Features

- User-friendly interface for flight queries
- Airport and airline autocomplete
- Date and time field for precise queries
- Prediction result cards display
- Modern styling with Tailwind CSS
- Skeleton loading while fetching results
- IATA code validation (e.g., GRU, GIG)

---

## Tech Stack

- React
- Vite
- Tailwind CSS
- Axios (for backend API consumption)
- Custom validations (IATA)
- Component-based architecture

---

## Prerequisites

- Node.js (recommended v16+)
- NPM or Yarn

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dragoscalin33/flight-on-time-ds.git
```

2. Navigate to the project folder:
```bash
cd front-end
```

3. Install dependencies:
```bash
npm install
```

4. Start the application:
```bash
npm run dev
```

The application will open automatically in the browser at: `http://localhost:5173`

---

## Project Structure

```
front-end/
├─ public/
├─ src/
│  ├─ components/        # Reusable components (FlightCard, SkeletonCard, etc.)
│  ├─ data/              # Static data (airports, airlines)
│  ├─ pages/             # Main pages (FlightSearch, Dashboard, etc.)
│  ├─ services/          # Axios client configuration
│  ├─ utils/             # Validations, helpers, utilities
│  ├─ App.jsx            # Route/UI entry point
│  └─ index.css          # Global styles
├─ package.json
├─ tailwind.config.js
├─ vite.config.js
└─ README.md
```

---

## Usage

1. Fill in the fields:
   - Airline
   - Origin (IATA code or airport name)
   - Destination (IATA code or airport name)
   - Flight date and time
2. Click **"Check flight"**.
3. View the result card with prediction and delay probability.

---

## Notes

- The interface is designed to work with the backend API.
- To connect to your backend, adjust the `baseURL` in `src/services/api.js`.

---

<details>
<summary><strong>Versao em Portugues / Portuguese Version</strong></summary>

O **Flight On Time — Dashboard** e a interface frontend que permite aos usuarios consultar previsoes de atraso de voos com base em origem, destino, companhia e data/hora.

Desenvolvido com **React + Vite + Tailwind CSS**, comunica-se com a API backend para prever atrasos. A interface e responsiva, interativa e conta com autocomplete para aeroportos e companhias aereas, alem de validacoes de IATA.

**Stack:** React, Vite, Tailwind CSS, Axios. **Requisitos:** Node.js 16+, NPM ou Yarn.

Para detalhes completos de instalacao e uso, consulte a versao em ingles acima.

</details>
