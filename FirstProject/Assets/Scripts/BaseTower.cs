using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class BaseTower : MonoBehaviour
{
    // Start is called before the first frame update
    public int fullHealth = 5;
    public int currentHealth = 5;
    
    public Slider ui;
    void Start()
    {
        Time.timeScale = 1.0f;
    }

    // Update is called once per frame
    void Update()
    {
        
        ui.value = currentHealth;
    }
}
