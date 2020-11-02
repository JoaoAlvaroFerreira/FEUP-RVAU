using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FaceCamera : MonoBehaviour
{

    public new Camera camera;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        var n = camera.transform.position;
        transform.rotation = Quaternion.LookRotation(n) * Quaternion.Euler(0, 0, 180);
    }
}
