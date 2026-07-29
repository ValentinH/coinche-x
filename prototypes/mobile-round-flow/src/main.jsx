import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const VARIANTS = {
  A: "Guidage pas à pas",
  B: "Tableau de manche",
  C: "Mode table",
};

const PHASES = ["setup", "contract", "play", "capture", "review", "result"];
const PHASE_LABELS = {
  setup: "Partie",
  contract: "Contrat",
  play: "Annonces",
  capture: "Plis",
  review: "Contrôle",
  result: "Score",
};

const SAMPLE_CARDS = [
  "A♥",
  "10♥",
  "V♥",
  "9♥",
  "A♣",
  "10♣",
  "R♣",
  "D♣",
  "A♦",
  "10♦",
  "V♦",
  "8♦",
  "A♠",
  "10♠",
  "R♠",
  "D♠",
  "9♣",
  "8♣",
  "7♦",
  "7♠",
];

const INITIAL_STATE = {
  phase: "setup",
  round: 1,
  players: ["Valentin", "Alice", "Nora", "Marc"],
  target: 2000,
  keeper: 0,
  scores: { a: 0, b: 0 },
  contract: {
    team: "a",
    value: "100",
    suit: "♥",
    multiplier: 1,
    declarer: 0,
  },
  announcements: { a: 20, b: 0 },
  belote: "a",
  dixDeDer: "a",
  capture: "pending",
  detectedCards: SAMPLE_CARDS,
  corrections: 0,
  result: null,
};

function App() {
  const [variant, setVariant] = useState(readVariant);
  const [game, setGame] = useState(INITIAL_STATE);

  useEffect(() => {
    const onPopState = () => setVariant(readVariant());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const changeVariant = (next) => {
    const url = new URL(window.location.href);
    url.searchParams.set("variant", next);
    window.history.replaceState({}, "", url);
    setVariant(next);
  };

  const api = useMemo(
    () => ({
      next() {
        setGame((current) => {
          const index = PHASES.indexOf(current.phase);
          return { ...current, phase: PHASES[Math.min(index + 1, PHASES.length - 1)] };
        });
      },
      go(phase) {
        setGame((current) => ({ ...current, phase }));
      },
      updatePlayer(index, value) {
        setGame((current) => ({
          ...current,
          players: current.players.map((player, playerIndex) =>
            playerIndex === index ? value : player,
          ),
        }));
      },
      patchContract(patch) {
        setGame((current) => ({
          ...current,
          contract: { ...current.contract, ...patch },
        }));
      },
      addAnnouncement(team, value) {
        setGame((current) => ({
          ...current,
          announcements: {
            ...current.announcements,
            [team]: current.announcements[team] + value,
          },
        }));
      },
      resetAnnouncements(team) {
        setGame((current) => ({
          ...current,
          announcements: { ...current.announcements, [team]: 0 },
        }));
      },
      setBelote(team) {
        setGame((current) => ({ ...current, belote: current.belote === team ? null : team }));
      },
      capture(mode) {
        setGame((current) => ({
          ...current,
          capture: mode,
          detectedCards: mode === "none" ? [] : SAMPLE_CARDS,
          phase: "review",
        }));
      },
      setDixDeDer(team) {
        setGame((current) => ({ ...current, dixDeDer: team }));
      },
      removeCard() {
        setGame((current) => ({
          ...current,
          detectedCards: current.detectedCards.slice(0, -1),
          corrections: current.corrections + 1,
        }));
      },
      addCard() {
        setGame((current) => ({
          ...current,
          detectedCards: [
            ...current.detectedCards,
            SAMPLE_CARDS[current.detectedCards.length % SAMPLE_CARDS.length],
          ],
          corrections: current.corrections + 1,
        }));
      },
      validate() {
        setGame((current) => ({
          ...current,
          phase: "result",
          result: {
            made: true,
            trickPoints: 92,
            delta: { a: 232, b: 70 },
          },
          scores: {
            a: current.scores.a + 232,
            b: current.scores.b + 70,
          },
        }));
      },
      nextRound() {
        setGame((current) => ({
          ...INITIAL_STATE,
          players: current.players,
          target: current.target,
          keeper: current.keeper,
          scores: current.scores,
          round: current.round + 1,
          phase: "contract",
        }));
      },
      restart() {
        setGame(INITIAL_STATE);
      },
      setTarget(target) {
        setGame((current) => ({ ...current, target }));
      },
      setKeeper(keeper) {
        setGame((current) => ({ ...current, keeper }));
      },
    }),
    [],
  );

  const props = { game, api };

  return (
    <main className={`prototype-root variant-${variant.toLowerCase()}`}>
      <PrototypeNotice />
      {variant === "A" && <VariantA {...props} />}
      {variant === "B" && <VariantB {...props} />}
      {variant === "C" && <VariantC {...props} />}
      <StateRibbon game={game} />
      {import.meta.env.DEV && (
        <PrototypeSwitcher
          current={variant}
          variants={Object.keys(VARIANTS)}
          onChange={changeVariant}
        />
      )}
    </main>
  );
}

function PrototypeNotice() {
  return <div className="prototype-notice">PROTOTYPE JETABLE · données simulées</div>;
}

function VariantA({ game, api }) {
  const phaseIndex = PHASES.indexOf(game.phase);

  return (
    <section className="phone guided-shell">
      <ScoreHeader game={game} compact />
      <div className="guided-progress" aria-label="Progression">
        {PHASES.map((phase, index) => (
          <button
            key={phase}
            className={index <= phaseIndex ? "is-active" : ""}
            onClick={() => api.go(phase)}
            title={PHASE_LABELS[phase]}
          >
            <span>{index + 1}</span>
          </button>
        ))}
      </div>
      <div className="guided-body">
        <p className="eyebrow">
          Manche {game.round} · Étape {phaseIndex + 1}/6
        </p>
        <h1>{PHASE_LABELS[game.phase]}</h1>
        {game.phase === "setup" && <SetupPanel game={game} api={api} />}
        {game.phase === "contract" && <ContractPanel game={game} api={api} />}
        {game.phase === "play" && <PlayPanel game={game} api={api} />}
        {game.phase === "capture" && <CapturePanel game={game} api={api} />}
        {game.phase === "review" && <ReviewPanel game={game} api={api} />}
        {game.phase === "result" && <ResultPanel game={game} api={api} />}
      </div>
    </section>
  );
}

function VariantB({ game, api }) {
  return (
    <section className="phone dashboard-shell">
      <div className="dashboard-hero">
        <div>
          <p className="eyebrow">Manche {game.round}</p>
          <h1>
            {game.scores.a} <span>—</span> {game.scores.b}
          </h1>
        </div>
        <div className="target-badge">Objectif {game.target}</div>
      </div>

      <div className="dashboard-grid">
        <section className="dash-card dash-contract">
          <div className="section-heading">
            <span>01</span>
            <div>
              <p>Contrat final</p>
              <strong>
                {game.contract.value} {game.contract.suit}
              </strong>
            </div>
          </div>
          <InlineContract game={game} api={api} />
        </section>

        <section className="dash-card dash-events">
          <div className="section-heading">
            <span>02</span>
            <div>
              <p>Prime gagnante</p>
              <strong>
                {game.announcements.a + (game.belote === "a" ? 20 : 0)} pts
              </strong>
            </div>
          </div>
          <div className="quick-row">
            {[20, 50, 100].map((value) => (
              <button key={value} onClick={() => api.addAnnouncement("a", value)}>
                +{value}
              </button>
            ))}
            <button
              className={game.belote === "a" ? "selected" : ""}
              onClick={() => api.setBelote("a")}
            >
              Belote
            </button>
          </div>
        </section>

        <section className="dash-card dash-capture">
          <div className="section-heading">
            <span>03</span>
            <div>
              <p>Plis de Nous</p>
              <strong>
                {game.capture === "photo"
                  ? `${game.detectedCards.length} cartes`
                  : "À compter"}
              </strong>
            </div>
          </div>
          {game.capture === "pending" ? (
            <div className="capture-split">
              <button className="camera-tile" onClick={() => api.capture("photo")}>
                <span>◎</span>
                Photographier
              </button>
              <button className="empty-tile" onClick={() => api.capture("none")}>
                Aucun pli
              </button>
            </div>
          ) : (
            <CompactReview game={game} api={api} />
          )}
        </section>
      </div>

      <button
        className="dashboard-commit"
        onClick={
          game.capture === "pending"
            ? () => api.capture("photo")
            : game.result
              ? api.nextRound
              : api.validate
        }
      >
        {game.result
          ? `Manche suivante · ${game.scores.a}–${game.scores.b}`
          : game.capture === "pending"
            ? "Compter les plis"
            : "Valider la manche"}
      </button>
    </section>
  );
}

function VariantC({ game, api }) {
  const teamAActive = game.contract.team === "a";
  return (
    <section className="phone table-shell">
      <div className="table-score">
        <button
          className={`team-side team-a ${teamAActive ? "taking" : ""}`}
          onClick={() => api.patchContract({ team: "a" })}
        >
          <span>NOUS</span>
          <strong>{game.scores.a}</strong>
          <small>{game.players[0]} + {game.players[2]}</small>
        </button>
        <div className="round-chip">M{game.round}</div>
        <button
          className={`team-side team-b ${!teamAActive ? "taking" : ""}`}
          onClick={() => api.patchContract({ team: "b" })}
        >
          <span>EUX</span>
          <strong>{game.scores.b}</strong>
          <small>{game.players[1]} + {game.players[3]}</small>
        </button>
      </div>

      <div className="table-focus">
        <p className="eyebrow">{PHASE_LABELS[game.phase]}</p>
        {game.phase === "setup" && (
          <>
            <h1>Qui tient le score ?</h1>
            <div className="player-orbit">
              {game.players.map((player, index) => (
                <button
                  key={index}
                  className={game.keeper === index ? "selected" : ""}
                  onClick={() => api.setKeeper(index)}
                >
                  {player}
                </button>
              ))}
            </div>
            <BigAction onClick={api.next}>Commencer</BigAction>
          </>
        )}
        {game.phase === "contract" && (
          <>
            <h1>
              {game.contract.value} <Suit suit={game.contract.suit} />
            </h1>
            <div className="contract-wheel">
              {["80", "90", "100", "110", "120", "Capot"].map((value) => (
                <button
                  key={value}
                  className={game.contract.value === value ? "selected" : ""}
                  onClick={() => api.patchContract({ value })}
                >
                  {value}
                </button>
              ))}
            </div>
            <div className="suit-row">
              {["♣", "♦", "♥", "♠", "SA", "TA"].map((suit) => (
                <button
                  key={suit}
                  className={game.contract.suit === suit ? "selected" : ""}
                  onClick={() => api.patchContract({ suit })}
                >
                  {suit}
                </button>
              ))}
            </div>
            <BigAction onClick={api.next}>Contrat posé</BigAction>
          </>
        )}
        {game.phase === "play" && (
          <>
            <h1>Un événement ?</h1>
            <div className="event-stack">
              <button onClick={() => api.addAnnouncement("a", 20)}>
                <span>Annonce gagnante</span>
                <strong>+20</strong>
              </button>
              <button
                className={game.belote === "a" ? "selected" : ""}
                onClick={() => api.setBelote("a")}
              >
                <span>Belote · Nous</span>
                <strong>+20</strong>
              </button>
            </div>
            <BigAction onClick={api.next}>Fin de manche</BigAction>
          </>
        )}
        {game.phase === "capture" && (
          <>
            <h1>Posez vos plis</h1>
            <p className="focus-copy">Un coin visible par carte, groupées par quatre.</p>
            <button className="giant-camera" onClick={() => api.capture("photo")}>
              <span>◎</span>
              Prendre la photo
            </button>
            <button className="text-action" onClick={() => api.capture("none")}>
              Nous n’avons fait aucun pli
            </button>
          </>
        )}
        {game.phase === "review" && (
          <>
            <h1>{game.detectedCards.length / 4 || 0} plis détectés</h1>
            <CardFan cards={game.detectedCards.slice(0, 8)} />
            <div className="review-toolbar">
              <button onClick={api.removeCard}>− carte</button>
              <button onClick={api.addCard}>+ carte</button>
            </div>
            <TeamToggle
              label="Dix de der"
              value={game.dixDeDer}
              onChange={api.setDixDeDer}
            />
            <BigAction onClick={api.validate}>Calculer</BigAction>
          </>
        )}
        {game.phase === "result" && (
          <>
            <div className="result-seal">✓</div>
            <h1>Contrat rempli</h1>
            <p className="focus-copy">
              Nous +232 · Eux +70
            </p>
            <BigAction onClick={api.nextRound}>Manche suivante</BigAction>
          </>
        )}
      </div>

      <nav className="table-dock">
        {PHASES.map((phase) => (
          <button
            key={phase}
            className={game.phase === phase ? "selected" : ""}
            onClick={() => api.go(phase)}
          >
            {PHASE_LABELS[phase]}
          </button>
        ))}
      </nav>
    </section>
  );
}

function SetupPanel({ game, api }) {
  return (
    <div className="panel-stack">
      <div className="team-editor">
        <p>Équipe Nous</p>
        {[0, 2].map((index) => (
          <input
            key={index}
            value={game.players[index]}
            onChange={(event) => api.updatePlayer(index, event.target.value)}
          />
        ))}
      </div>
      <div className="team-editor">
        <p>Équipe Eux</p>
        {[1, 3].map((index) => (
          <input
            key={index}
            value={game.players[index]}
            onChange={(event) => api.updatePlayer(index, event.target.value)}
          />
        ))}
      </div>
      <label className="field">
        Responsable du score
        <select value={game.keeper} onChange={(event) => api.setKeeper(Number(event.target.value))}>
          {game.players.map((player, index) => (
            <option value={index} key={index}>
              {player}
            </option>
          ))}
        </select>
      </label>
      <div>
        <p className="field-label">Score cible</p>
        <Segmented
          values={[1500, 2000, 3000]}
          selected={game.target}
          onSelect={api.setTarget}
        />
      </div>
      <PrimaryAction onClick={api.next}>Créer la partie</PrimaryAction>
    </div>
  );
}

function ContractPanel({ game, api }) {
  return (
    <div className="panel-stack">
      <TeamToggle
        label="Équipe preneuse"
        value={game.contract.team}
        onChange={(team) => api.patchContract({ team })}
      />
      <div>
        <p className="field-label">Valeur</p>
        <div className="value-grid">
          {["80", "90", "100", "110", "120", "130", "140", "150", "160"].map(
            (value) => (
              <button
                key={value}
                className={game.contract.value === value ? "selected" : ""}
                onClick={() => api.patchContract({ value })}
              >
                {value}
              </button>
            ),
          )}
        </div>
        <div className="special-row">
          {["Capot", "Générale"].map((value) => (
            <button
              key={value}
              className={game.contract.value === value ? "selected" : ""}
              onClick={() => api.patchContract({ value })}
            >
              {value}
            </button>
          ))}
        </div>
      </div>
      <div>
        <p className="field-label">Atout</p>
        <div className="suit-picker">
          {["♣", "♦", "♥", "♠", "SA", "TA"].map((suit) => (
            <button
              key={suit}
              className={game.contract.suit === suit ? "selected" : ""}
              onClick={() => api.patchContract({ suit })}
            >
              <Suit suit={suit} />
            </button>
          ))}
        </div>
      </div>
      <div>
        <p className="field-label">Enjeu</p>
        <Segmented
          values={[
            { label: "Normal", value: 1 },
            { label: "Coinché", value: 2 },
            { label: "Surcoinché", value: 4 },
          ]}
          selected={game.contract.multiplier}
          onSelect={(multiplier) => api.patchContract({ multiplier })}
        />
      </div>
      <PrimaryAction onClick={api.next}>
        Valider {game.contract.value} {game.contract.suit}
      </PrimaryAction>
    </div>
  );
}

function PlayPanel({ game, api }) {
  return (
    <div className="panel-stack">
      <div className="live-contract">
        <span>Contrat en cours</span>
        <strong>
          {game.contract.value} <Suit suit={game.contract.suit} />
        </strong>
        <small>{game.contract.team === "a" ? "Nous prenons" : "Eux prennent"}</small>
      </div>
      {["a", "b"].map((team) => (
        <section className="announcement-team" key={team}>
          <div>
            <span>{team === "a" ? "Nous" : "Eux"}</span>
            <strong>{game.announcements[team]} pts</strong>
          </div>
          <div className="quick-row">
            {[20, 50, 100, 150, 200].map((value) => (
              <button key={value} onClick={() => api.addAnnouncement(team, value)}>
                +{value}
              </button>
            ))}
          </div>
          <div className="announcement-footer">
            <button
              className={game.belote === team ? "selected" : ""}
              onClick={() => api.setBelote(team)}
            >
              Belote
            </button>
            <button onClick={() => api.resetAnnouncements(team)}>Effacer</button>
          </div>
        </section>
      ))}
      <PrimaryAction onClick={api.next}>La manche est terminée</PrimaryAction>
    </div>
  );
}

function CapturePanel({ api }) {
  return (
    <div className="capture-panel">
      <div className="camera-guide">
        <span className="corner top-left" />
        <span className="corner top-right" />
        <span className="corner bottom-left" />
        <span className="corner bottom-right" />
        <div>
          <strong>Étalez vos plis</strong>
          <p>Cartes par quatre · un coin visible</p>
        </div>
      </div>
      <PrimaryAction onClick={() => api.capture("photo")}>◎ Prendre la photo</PrimaryAction>
      <button className="secondary-action" onClick={() => api.capture("none")}>
        Aucun pli pour Nous
      </button>
    </div>
  );
}

function ReviewPanel({ game, api }) {
  const isCoherent = game.detectedCards.length % 4 === 0;
  return (
    <div className="panel-stack">
      <div className={`recognition-status ${isCoherent ? "success" : "warning"}`}>
        <span>{isCoherent ? "✓" : "!"}</span>
        <div>
          <strong>
            {game.detectedCards.length} cartes · {Math.floor(game.detectedCards.length / 4)} plis
          </strong>
          <p>{isCoherent ? "Aucune incohérence détectée" : "Le total doit être multiple de quatre"}</p>
        </div>
      </div>
      <CardGrid cards={game.detectedCards} />
      <div className="correction-row">
        <button onClick={api.removeCard}>− Retirer</button>
        <span>{game.corrections} correction{game.corrections > 1 ? "s" : ""}</span>
        <button onClick={api.addCard}>+ Ajouter</button>
      </div>
      <TeamToggle label="Dix de der" value={game.dixDeDer} onChange={api.setDixDeDer} />
      <PrimaryAction onClick={api.validate} disabled={!isCoherent}>
        Valider et calculer
      </PrimaryAction>
    </div>
  );
}

function ResultPanel({ game, api }) {
  return (
    <div className="result-panel">
      <div className="result-icon">✓</div>
      <p className="eyebrow">Manche enregistrée</p>
      <h2>Contrat rempli</h2>
      <div className="result-score">
        <div>
          <span>Nous</span>
          <strong>+{game.result?.delta.a ?? 232}</strong>
        </div>
        <div>
          <span>Eux</span>
          <strong>+{game.result?.delta.b ?? 70}</strong>
        </div>
      </div>
      <dl className="score-detail">
        <div><dt>Plis</dt><dd>92</dd></div>
        <div><dt>Annonce + Belote</dt><dd>40</dd></div>
        <div><dt>Contrat</dt><dd>100</dd></div>
      </dl>
      <PrimaryAction onClick={api.nextRound}>Manche suivante</PrimaryAction>
      <button className="secondary-action" onClick={() => api.go("review")}>
        Corriger cette manche
      </button>
    </div>
  );
}

function InlineContract({ game, api }) {
  return (
    <>
      <div className="dash-values">
        {["80", "100", "120", "Capot"].map((value) => (
          <button
            key={value}
            className={game.contract.value === value ? "selected" : ""}
            onClick={() => api.patchContract({ value })}
          >
            {value}
          </button>
        ))}
      </div>
      <div className="dash-suits">
        {["♣", "♦", "♥", "♠", "SA", "TA"].map((suit) => (
          <button
            key={suit}
            className={game.contract.suit === suit ? "selected" : ""}
            onClick={() => api.patchContract({ suit })}
          >
            {suit}
          </button>
        ))}
      </div>
    </>
  );
}

function CompactReview({ game, api }) {
  return (
    <div className="compact-review">
      <div className="review-summary">
        <span className="status-dot" />
        <strong>{game.detectedCards.length} cartes reconnues</strong>
        <span>{game.corrections ? `${game.corrections} corrigée(s)` : "Prêt"}</span>
      </div>
      <CardFan cards={game.detectedCards.slice(0, 6)} />
      <div className="quick-row">
        <button onClick={api.removeCard}>− carte</button>
        <button onClick={api.addCard}>+ carte</button>
        <button onClick={() => api.go("capture")}>Reprendre</button>
      </div>
    </div>
  );
}

function ScoreHeader({ game, compact = false }) {
  return (
    <header className={`score-header ${compact ? "compact" : ""}`}>
      <div>
        <span>NOUS</span>
        <strong>{game.scores.a}</strong>
      </div>
      <button title="Reprendre la partie">
        <small>Objectif {game.target}</small>
        <b>Coinche X</b>
      </button>
      <div>
        <span>EUX</span>
        <strong>{game.scores.b}</strong>
      </div>
    </header>
  );
}

function TeamToggle({ label, value, onChange }) {
  return (
    <div>
      <p className="field-label">{label}</p>
      <div className="team-toggle">
        <button className={value === "a" ? "selected" : ""} onClick={() => onChange("a")}>
          Nous
        </button>
        <button className={value === "b" ? "selected" : ""} onClick={() => onChange("b")}>
          Eux
        </button>
      </div>
    </div>
  );
}

function Segmented({ values, selected, onSelect }) {
  return (
    <div className="segmented">
      {values.map((item) => {
        const value = typeof item === "object" ? item.value : item;
        const label = typeof item === "object" ? item.label : item;
        return (
          <button
            key={value}
            className={selected === value ? "selected" : ""}
            onClick={() => onSelect(value)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

function Suit({ suit }) {
  const isRed = suit === "♥" || suit === "♦";
  return <span className={isRed ? "red-suit" : ""}>{suit}</span>;
}

function PrimaryAction({ children, ...props }) {
  return (
    <button className="primary-action" {...props}>
      {children}
    </button>
  );
}

function BigAction({ children, ...props }) {
  return (
    <button className="big-action" {...props}>
      {children}
      <span>→</span>
    </button>
  );
}

function CardGrid({ cards }) {
  if (cards.length === 0) {
    return <div className="empty-cards">Aucune carte · aucun pli</div>;
  }
  return (
    <div className="card-grid">
      {cards.map((card, index) => (
        <MiniCard card={card} key={`${card}-${index}`} />
      ))}
    </div>
  );
}

function CardFan({ cards }) {
  if (cards.length === 0) return null;
  return (
    <div className="card-fan">
      {cards.map((card, index) => (
        <MiniCard card={card} key={`${card}-${index}`} style={{ "--index": index }} />
      ))}
    </div>
  );
}

function MiniCard({ card, style }) {
  const suit = card.slice(-1);
  const rank = card.slice(0, -1);
  return (
    <span className="mini-card" style={style}>
      <b className={suit === "♥" || suit === "♦" ? "red-suit" : ""}>
        {rank}
        <i>{suit}</i>
      </b>
    </span>
  );
}

function StateRibbon({ game }) {
  return (
    <div className="state-ribbon">
      <span>ÉTAT</span>
      M{game.round} · {PHASE_LABELS[game.phase]} · {game.contract.value}
      {game.contract.suit} · {game.announcements.a + game.announcements.b} annonce ·{" "}
      {game.detectedCards.length} cartes · {game.corrections} correction
    </div>
  );
}

function PrototypeSwitcher({ current, variants, onChange }) {
  useEffect(() => {
    const onKeyDown = (event) => {
      const target = event.target;
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement ||
        target?.isContentEditable
      ) {
        return;
      }
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const index = variants.indexOf(current);
      const direction = event.key === "ArrowRight" ? 1 : -1;
      onChange(variants[(index + direction + variants.length) % variants.length]);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [current, onChange, variants]);

  const cycle = (direction) => {
    const index = variants.indexOf(current);
    onChange(variants[(index + direction + variants.length) % variants.length]);
  };

  return (
    <nav className="prototype-switcher" aria-label="Variantes du prototype">
      <button onClick={() => cycle(-1)} aria-label="Variante précédente">←</button>
      <div>
        <small>VARIANTE</small>
        <strong>{current} — {VARIANTS[current]}</strong>
      </div>
      <button onClick={() => cycle(1)} aria-label="Variante suivante">→</button>
    </nav>
  );
}

function readVariant() {
  const candidate = new URLSearchParams(window.location.search).get("variant")?.toUpperCase();
  return VARIANTS[candidate] ? candidate : "A";
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
