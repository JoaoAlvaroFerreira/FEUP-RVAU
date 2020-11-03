using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;
using Vuforia;
public class GameOverMenu : MonoBehaviour
{
    public GameObject game;
    public TextMeshProUGUI text;
    private void Start()
    {
        Debug.Log("Start");
        text.text = "High Score: " + game.GetComponent<TakeDamage>().timeTotal;
    }
    public void quitToMenu()
    {

        SceneManager.LoadScene("MenuScene");

    }
    public void restartGame()
    {

        SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);


        Debug.Log("Time ReStart:" + Time.timeScale);
    }
}
