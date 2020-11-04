using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class MenuScript : MonoBehaviour
{

    public static float difficulty;

    private void Start(){
        difficulty=1f;
    }

    public void PlayGame()
    {
        SceneManager.LoadScene("GameScene");
       
    }

    public void ChangeDifficulty(int s){
       
        switch(s){
            case 0:
            difficulty = 0.2f;
            Debug.Log("Changed to Easy");
            break;
            case 1:
            difficulty = 1f;
            Debug.Log("Changed to Normal");
            break;
            case 2:
            difficulty = 5f;
            Debug.Log("Changed to Hard");
            break;
        }
    }

    public void QuitGame()
    {
        Application.Quit();
    }
}
