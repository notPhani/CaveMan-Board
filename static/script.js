document.getElementById("loginForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const username = document.getElementById("User-name").value;
  const password = document.getElementById("User-password").value;

  const payload = {
    username,
    password
  };

  try {
    const res = await fetch("http://localhost:8000/api/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.success) {
      localStorage.setItem("cavemanLoggedIn", "true");
      localStorage.setItem("cavemanUUID", data.uuid);

      // Hide login and show board
      document.getElementById("login-form").style.display = "none";
      document.getElementById("board").style.zIndex = "10";

      renderBoard(); // Your board rendering function
    } else {
      if (data.error === "wrong_pass") {
        alert("🪓 Wrong password, ooga!");
      } else if (data.error === "no_user") {
        alert("👤 User not found. Wanna sign in?");
      }
    }

  } catch (err) {
    console.error("Login error:", err);
    alert("Something broke in the cave.");
  }
});
window.addEventListener("DOMContentLoaded", () => {
  const loggedIn = localStorage.getItem("cavemanLoggedIn");

  if (loggedIn === "true") {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("board").style.zIndex = "10";
    renderBoard(); // Auto-show board
  }
});
