/* PREDICTION DEMO */
function predict() {
  const driver = document.getElementById("driver").value || "Selected driver";
  const team = document.getElementById("constructor").value || "selected team";
  const race = document.getElementById("race").value || "selected race";

  const results = ["P1 🏆", "P2 🔥", "P3 ⚡", "P4 🚀", "P5 🏁"];
  const confidence = Math.floor(Math.random() * 18) + 78;
  const randomResult = results[Math.floor(Math.random() * results.length)];

  const resultBox = document.getElementById("result");

  resultBox.innerHTML = `
    <span class="text-white">NEURAL OUTPUT READY</span><br><br>
    ${driver} with ${team}<br>
    Track: ${race}<br><br>
    <span class="text-white">Predicted Result:</span> ${randomResult}<br>
    <span class="text-white">Confidence:</span> ${confidence}%
  `;

  resultBox.classList.add("result-pulse");

  setTimeout(() => {
    resultBox.classList.remove("result-pulse");
  }, 900);
}

/* SCROLL REVEAL */
const revealElements = document.querySelectorAll(".reveal");

function revealOnScroll() {
  revealElements.forEach((element) => {
    const elementTop = element.getBoundingClientRect().top;
    const windowHeight = window.innerHeight;

    if (elementTop < windowHeight - 120) {
      element.classList.add("active");
    }
  });
}

window.addEventListener("scroll", revealOnScroll);
window.addEventListener("load", revealOnScroll);

/* HERO MOUSE GLOW MOVEMENT */
const heroGlow = document.querySelector(".hero-glow");

document.addEventListener("mousemove", (e) => {
  if (!heroGlow) return;

  const x = (e.clientX / window.innerWidth - 0.5) * 80;
  const y = (e.clientY / window.innerHeight - 0.5) * 50;

  heroGlow.style.transform = `translate(${x}px, ${y}px)`;
});

/* CYBER TYPING EFFECT */
const title = document.querySelector(".neon-text");

if (title) {
  title.style.opacity = "0";

  setTimeout(() => {
    title.style.opacity = "1";
    title.style.transition = "opacity 1s ease";
  }, 400);
}