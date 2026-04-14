import './App.css'
import Grain from './Grain'
import PlanetDisk from './PlanetDisk'

function App() {
  return (
    <>
      <Grain />

      {/* Decorative background — hidden from assistive tech */}
      <div className="bg" aria-hidden="true">
        <div className="bg__blob--1" />
        <div className="bg__blob--2" />
        <div className="bg__blob--3" />
        <div className="bg__line bg__line--1" />
        <div className="bg__line bg__line--2" />
        <div className="bg__line bg__line--3" />
        <div className="bg__line bg__line--4" />
      </div>

      <div className="page">
        <div className="content">
          {/* Portal — decorative, screen readers skip to the heading */}
          <div className="portal" aria-hidden="true">
            <div className="portal__layer portal__glow" />
            <div className="portal__layer portal__wisps" />
            <div className="portal__layer portal__swirl-outer" />
            <div className="portal__layer portal__swirl-mid" />
            <div className="portal__layer portal__swirl-inner" />
            <div className="portal__layer portal__corona" />
            <div className="portal__layer portal__corona-flare" />
            <div className="portal__layer portal__horizon" />
            <PlanetDisk />
          </div>
          <h1 className="title">HYPERION</h1>
        </div>

        <nav className="contact" aria-label="Contact links">
          <a href="mailto:contact@hyperion.com">contact@hyperion.com</a>
          <span className="sep" aria-hidden="true">&#9679;</span>
          <a
            href="https://www.linkedin.com/in/jonah-elliott-87771024a/"
            target="_blank"
            rel="noopener noreferrer"
          >
            LinkedIn
          </a>
        </nav>
      </div>
    </>
  )
}

export default App
