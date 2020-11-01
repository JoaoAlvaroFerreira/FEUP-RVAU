using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class TakeDamage : MonoBehaviour
{
    
    // Start is called before the first frame update

    
    // Start is called before the first frame update
    public int fullHealth = 40;
    public int currentHealth = 40;
    public Slider ui;   
    void Update()
    {
        ui.value = currentHealth;
    }
    // Update is called once per frame

     void OnTriggerEnter(Collider collision)
    {
       
        //Check for a match with the specified name on any GameObject that collides with your GameObject
        if (collision.gameObject.name == "Enemy Hand")
        {
            //If the GameObject's name matches the one you suggest, output this message in the console
            currentHealth = currentHealth - 1;
        }

       
    }
}
