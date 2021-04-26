package ui.home;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.fragment.app.FragmentManager;
import androidx.fragment.app.FragmentTransaction;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.NavigationUI;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.cfm.recommend.Recommender;
import com.example.cfm.R;
import com.squareup.picasso.Picasso;

import spotify_framework.Song;
import spotify_framework.SongService;
import ui.playback.SongFragment;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static java.lang.Thread.sleep;

public class HomeFragment extends Fragment {
    public SongService songService;
    private RecyclerView recyclerView;
    private SongsAdapter adapter;
    private Recommender recommender;
    private Button radioStart;



    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {
        View root = inflater.inflate(R.layout.fragment_home, container, false);
        radioStart = (Button) root.findViewById(R.id.begin_listening);
        songService = new SongService(getActivity());
        songService.getRecentlyPlayed(() -> {
            ArrayList<Song> recents = songService.getPlaylist();
            adapter = new SongsAdapter(getActivity(), recents);
            adapter.notifyDataSetChanged();
            recyclerView = (RecyclerView) getActivity().findViewById(R.id.listening_history);
            recyclerView.setAdapter(adapter);
            recyclerView.setLayoutManager(new LinearLayoutManager(getActivity()));
        });
        radioStart.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Ask the recommender for a set of songs to place in queue
                //Send most recent location and recent songs w/ neutral rating to DataStore
                /*
                This is for testing to make sure song information is being transferred to the
                playback fragment and that the playback fragment plays the correct songs
                 */
                songService = new SongService(getActivity());
                songService.getRecentlyPlayed(() -> {
                    ArrayList<Song> songUris = songService.getPlaylist();
                    System.out.println("Song URI size:: " + songUris.size());
                    SongFragment fragment = new SongFragment();
                    Bundle queue = new Bundle();
                    queue.putSerializable("songs", songUris);

                    ImageView album = getActivity().findViewById(R.id.album_art);
                    Picasso.get().load(songUris.get(0).getImages().get(0)).into(album);

                    NavController navController = Navigation.findNavController(getActivity(), R.id.nav_host_fragment);
                    navController.navigate(R.id.song_dashboard);
                });

            }
        });

        return root;
    }

    @Override
    public void onActivityCreated(Bundle savedInstanceState) {
        super.onActivityCreated(savedInstanceState);

    }
}