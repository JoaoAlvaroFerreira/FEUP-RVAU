using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
public class GameOverMenu : MonoBehaviour
{
   
    public void quitToMenu(){
         
        SceneManager.LoadScene("MenuScene");
    
    }
    public void restartGame(){
        SceneManager.LoadScene (SceneManager.GetActiveScene ().buildIndex);
        Time.timeScale = 1.0f;
    }
}
