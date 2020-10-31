using DigitalRuby.RainMaker;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using UnityEngine;
using UnityEngine.Networking;

public class Weather : MonoBehaviour
{
    private ExternalIPResponse currentIP;
    private ExternalLocationInformation externalLocationInformation;
    private ExternalWeatherInformation externalWeatherInformation;

    private bool rainStatus = false;
    private float rainInten = 0f;

    public GameObject rain;

    void Start()
    {
        StartCoroutine(SendRequest());
        rain.SetActive(false);
        rain.GetComponent<RainScript>().RainIntensity = 0f;
    }

    // Update is called once per frame
    void Update()
    {
    }
    [Serializable]
    private class ExternalIPResponse
    {
        public string ip = null;
    }
    [Serializable]
    private class ExternalLocationInformation
    {
        public string geoplugin_request = null;
        public int geoplugin_status = 0;
        public string geoplugin_delay = null;
        public string geoplugin_credit = null;
        public string geoplugin_city = null;
        public string geoplugin_region = null;
        public string geoplugin_regionCode = null;
        public string geoplugin_regionName = null;
        public string geoplugin_areaCode = null;
        public string geoplugin_dmaCode = null;
        public string geoplugin_countryCode = null;
        public string geoplugin_countryName = null;
        public int geoplugin_inEU = 0;
        public int geoplugin_euVATrate = 0;
        public string geoplugin_continentCode = null;
        public string geoplugin_continentName = null;
        public string geoplugin_latitude = null;
        public string geoplugin_longitude = null;
        public string geoplugin_locationAccuracyRadius = null;
        public string geoplugin_timezone = null;
        public string geoplugin_currencyCode = null;
        public string geoplugin_currencySymbol = null;
        public string geoplugin_currencySymbol_UTF8 = null;
        public float geoplugin_currencyConverter = 0;
    }
    [Serializable]
    private class ExternalWeatherInformation
    {
        public ExternalCoord coord = null;
        public ExternalWeather[] weather = null;
        public ExternalMain main = null;
        public int visibility = 0;
        public ExternalWind wind = null;
        public ExternalClouds clouds = null;
        public int dt = 0;
        public ExternalSys sys = null;
        public int timezone = 0;
        public int id = 0;
        public string name = null;
        public int cod = 200;
    }
    [Serializable]
    private class ExternalCoord
    {
        public float lon = 0;
        public float lat = 0;
    }
    [Serializable]
    private class ExternalWeather
    {
        public int id = 0;
        public string main = null;
        public string description = null;
        public string icon = null;
    }
    [Serializable]
    private class ExternalMain
    {
        public float temp = 0;
        public float feels_like = 0;
        public float temp_min = 0;
        public float temp_max = 0;
        public int pressure = 0;
        public int humidity = 0;
        public string main = null;
        public string description = null;
        public string icon = null;
    }
    [Serializable]
    private class ExternalWind
    {
        public float speed = 0;
        public int deg = 0;
    }
    [Serializable]
    private class ExternalClouds
    {
        public int all = 0;
    }
    [Serializable]
    private class ExternalSys
    {
        public int type = 0;
        public int id = 0;
        public float message = 0;
        public string country = null;
        public int sunrise = 0;
        public int sunset = 0;
    }

    IEnumerator SendRequest()
    {
        UnityWebRequest webRequest = UnityWebRequest.Get("https://api.ipify.org?format=json");
        yield return webRequest.SendWebRequest();
        if (!webRequest.isNetworkError)
        {
            currentIP = JsonUtility.FromJson<ExternalIPResponse>(webRequest.downloadHandler.text);
            webRequest = UnityWebRequest.Get("http://www.geoplugin.net/json.gp?ip=" + currentIP);
            yield return webRequest.SendWebRequest();
            if (!webRequest.isNetworkError)
            {
                externalLocationInformation = JsonUtility.FromJson<ExternalLocationInformation>(webRequest.downloadHandler.text);
                webRequest = UnityWebRequest.Get("http://api.openweathermap.org/data/2.5/weather?q=" + "London" + "&appid=d67b3b963691d6ea4b8f646ac3fb3337");
                yield return webRequest.SendWebRequest();
                Debug.Log("WE ARE IN:");
                Debug.Log(externalLocationInformation.geoplugin_city);
                if (!webRequest.isNetworkError)
                {
                    externalWeatherInformation = JsonUtility.FromJson<ExternalWeatherInformation>(webRequest.downloadHandler.text);
                    if (externalWeatherInformation.cod == 200 && externalWeatherInformation.weather.Length >= 1)
                    {
                        foreach (ExternalWeather weather in externalWeatherInformation.weather)
                        {
                            if (weather.id >= 200 && weather.id < 300)
                            {
                                rainStatus = true;
                                switch (weather.id)
                                {
                                    case 200:
                                        rainInten = 0.25f;
                                        break;
                                    case 201:
                                        rainInten = 0.5f;
                                        break;
                                    case 202:
                                        rainInten = 1f;
                                        break;
                                    case 210:
                                        rainInten = 0.5f;
                                        break;
                                    case 211:
                                        rainInten = 0.75f;
                                        break;
                                    case 212:
                                        rainInten = 1f;
                                        break;
                                    case 221:
                                        rainInten = 1f;
                                        break;
                                    case 230:
                                        rainInten = 0.5f;
                                        break;
                                    case 231:
                                        rainInten = 0.6f;
                                        break;
                                    case 232:
                                        rainInten = 0.7f;
                                        break;
                                    default:
                                        break;
                                }
                            }
                            if (weather.id >= 300 && weather.id < 400)
                            {
                                rainStatus = true;
                                rainInten = 0.05f;
                            }
                            if (weather.id >= 500 && weather.id < 600)
                            {
                                rainStatus = true;
                                switch (weather.id)
                                {
                                    case 500:
                                        rainInten = 0.2f;
                                        break;
                                    case 501:
                                        rainInten = 0.4f;
                                        break;
                                    case 502:
                                        rainInten = 0.6f;
                                        break;
                                    case 503:
                                        rainInten = 0.8f;
                                        break;
                                    case 504:
                                        rainInten = 1f;
                                        break;
                                    case 511:
                                        rainInten = 0.4f;
                                        break;
                                    case 520:
                                        rainInten = 0.25f;
                                        break;
                                    case 521:
                                        rainInten = 0.5f;
                                        break;
                                    case 522:
                                        rainInten = 1f;
                                        break;
                                    case 531:
                                        rainInten = 1f;
                                        break;
                                    default:
                                        break;
                                }
                            }
                            if (weather.id >= 600 && weather.id < 700)
                            {
                                rainStatus = true;
                                switch (weather.id)
                                {
                                    case 600:
                                        rainInten = 0.25f;
                                        break;
                                    case 601:
                                        rainInten = 0.5f;
                                        break;
                                    case 602:
                                        rainInten = 1f;
                                        break;
                                    case 611:
                                        rainInten = 0.1f;
                                        break;
                                    case 612:
                                        rainInten = 0.2f;
                                        break;
                                    case 613:
                                        rainInten = 0.4f;
                                        break;
                                    case 615:
                                        rainInten = 0.5f;
                                        break;
                                    case 616:
                                        rainInten = 0.8f;
                                        break;
                                    case 620:
                                        rainInten = 0.25f;
                                        break;
                                    case 621:
                                        rainInten = 0.5f;
                                        break;
                                    case 622:
                                        rainInten = 1f;
                                        break;
                                    default:
                                        break;
                                }
                            }
                            if (weather.id >= 800 && weather.id < 900)
                            {
                                rainStatus = false;
                            }
                        }
                        rain.SetActive(rainStatus);
                        rain.GetComponent<RainScript>().RainIntensity = rainInten;
                        Debug.Log("Weather set");
                    }
                }
            }
        }
    }
}
