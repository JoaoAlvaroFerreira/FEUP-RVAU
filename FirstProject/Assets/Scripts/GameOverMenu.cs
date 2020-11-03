using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;
public class GameOverMenu : MonoBehaviour
{
   public GameObject game;
   public TextMeshProUGUI text;
    private void Start() {
        Debug.Log("Start");
        text.text = "High Score: " +  game.GetComponent<TakeDamage>().timeTotal;
    }
    public void quitToMenu(){
         
        SceneManager.LoadScene("MenuScene");
    
    }
    public void restartGame(){
        Time.timeScale = 1.0f;
        SceneManager.LoadScene (SceneManager.GetActiveScene ().buildIndex);
        Time.timeScale = 1.0f;
    }
}
