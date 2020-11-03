using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SpaceshipController : MonoBehaviour
{

    public float rotationSpeed = 5f;

    public GameObject bullet;
    public new Camera camera;

    public float below_spaceship = 2f;

    public AudioSource audio;

    private float offset = 0.05f;

    // Start is called before the first frame update
    void Start()
    {

    }

    // Update is called once per frame
    void Update()
    {
        transform.RotateAround(new Vector3(0, 0, 0), Vector3.up, rotationSpeed * Time.deltaTime);
    }

    public void FireBullet()
    {
        RaycastHit[] hits = Physics.RaycastAll(camera.transform.position, camera.transform.forward);
        Vector3 point = new Vector3(0,0,0);
        bool enemy_detected = false;
        for (int i =0; i< hits.Length;i++)
        {
            if(hits[i].collider.gameObject.tag == "Floor" && !enemy_detected)
            {
                point = new Vector3((float)(hits[i].point.x - ((hits[i].point.x - camera.transform.position.x) * offset)), hits[i].point.y, (float)(hits[i].point.z - ((hits[i].point.z - camera.transform.position.z) * offset)));
            }
            else if (hits[i].collider.gameObject.tag == "Enemy")
            {
                enemy_detected = true;
                point = hits[i].collider.gameObject.transform.position;
            }
        }
        Vector3 spawn_point = new Vector3(transform.position.x, transform.position.y - below_spaceship, transform.position.z);
        GameObject bullet_object = Instantiate(bullet, spawn_point, Quaternion.LookRotation((point - spawn_point).normalized));
        audio.Play();
    }
}
