package ui.playback;

import android.content.SharedPreferences;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.os.Build;
import android.os.Bundle;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.cfm.R;
import com.google.gson.Gson;
import com.squareup.picasso.Picasso;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Queue;
import java.util.Set;
import java.util.stream.Collectors;

import spotify_framework.PlaybackService;
import spotify_framework.Song;


public class SongFragment extends Fragment {
    public static ImageView albumArt;
    public static TextView artists;
    public static TextView albumName;
    public static TextView songName;

    public Button playButton;
    public Button nextButton;
    public Button prevButton;
    private boolean isPaused = false;
    public ArrayList<Song> songQueue = new ArrayList<Song>();
    private int songPos = 0;
    private SharedPreferences preferences;
    private PlaybackService playbackService;


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        }


    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        // Inflate the layout for this fragment
        preferences = getActivity().getSharedPreferences("SPOTIFY", 0);
        View root = inflater.inflate(R.layout.fragment_song, container, false);
        albumArt = root.findViewById(R.id.album_art);

        playButton = root.findViewById(R.id.play);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(isPaused) {
                    isPaused = false;
                    playSong();
                    playButton.setText("Pause");
                }
                else {
                    pauseSong();
                    isPaused = true;
                    playButton.setText("Play");
                }
            }
        });
        nextButton = root.findViewById(R.id.next);
        nextButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                skipSong();
                songPos ++;
                setSongLayout(root, songQueue.get(songPos));
            }
        });
        prevButton = root.findViewById(R.id.prev);
        prevButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                playLast();
            }
        });

        playbackService = new PlaybackService(getActivity());

        Gson gson = new Gson();
        Set<String> songsJson = preferences.getStringSet("songs", null);
        for(Iterator<String> i = songsJson.iterator(); i.hasNext();) {
            String song = i.next();
            try {
                JSONObject object = new JSONObject(song);
                Song s = gson.fromJson(song, Song.class);
                JSONArray artists = object.optJSONArray("artistsList");
                JSONArray albumArt = object.optJSONArray("albumImages");
                populateSongInfo(s, artists, albumArt);
                songQueue.add(s);
            } catch (JSONException e) {
                e.printStackTrace();
            }
        }

        if(!hasImage(albumArt)) {
            setSongLayout(root,songQueue.get(songPos));
            initializeQueue();
        }
        playSong();
        return root;
    }


    @Override
    public void onActivityCreated(@Nullable Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);
    }

    private boolean hasImage(@NonNull ImageView view) {
        Drawable drawable = view.getDrawable();
        boolean hasImage = (drawable != null);

        if (hasImage && (drawable instanceof BitmapDrawable)) {
            hasImage = ((BitmapDrawable)drawable).getBitmap() != null;
        }

        return hasImage;
    }

    private void populateSongInfo(Song s, JSONArray artists, JSONArray albumImages) throws JSONException {
        for(int i = 0; i < artists.length(); i++) {
            s.setArtist(artists.getString(i));
        }

        for(int j = 0; j < albumImages.length(); j++) {
            s.setImage(albumImages.getString(j));
        }
    }

    private void setSongLayout(View view, Song song) {
        artists = view.findViewById(R.id.song_artist);
        albumName = view.findViewById(R.id.song_album);
        songName = view.findViewById(R.id.song_title);

        Picasso.get().load(song.getImages().get(0)).into(albumArt);
        artists.setText(song.artistsToString(Integer.MAX_VALUE));
        albumName.setText(song.getAlbum());
        songName.setText(song.getName());

    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    private void initializeQueue() {
        List<Song> songs = songQueue.stream().collect(Collectors.toList());
        songs.forEach((song) ->
                playbackService.findDevice(()-> {
                    playbackService.addToQueue(song);
                })
        );
    }

    private void playSong() {
        playbackService.findDevice(() -> {
            playbackService.play();
        });
    }

    private void pauseSong() {
        playbackService.findDevice(() -> {
            playbackService.pause();
        });
    }

    private void skipSong() {
        playbackService.findDevice(() -> {
            playbackService.next();
        });
    }

    private void playLast() {
        playbackService.findDevice(() -> {
            playbackService.prev();
        });
    }
}